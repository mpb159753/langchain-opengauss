import json
import re
from hashlib import md5
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Pattern, Tuple, Union

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore
from langchain_community.graphs.age_graph import AGEGraph, AGEQueryException
from langchain_opengauss import OpenGaussSettings

class openGaussAGEGraph(AGEGraph):
    """
    openGauss AGE wrapper for graph operations.
    Args:
        graph_name (str): the name of the graph to connect to or create
        conf (Dict[str, Any]): the pgsql connection config passed directly
            to psycopg2.connect
        create (bool): if True and graph doesn't exist, attempt to create it
    """

    # precompiled regex for checking chars in graph labels
    label_regex: Pattern = re.compile("[^0-9a-zA-Z\u4e00-\u9fff]+")

    def __init__(
        self, graph_name: str, conf: OpenGaussSettings, create: bool = True, **kwargs
    ) -> None:
        """Create a new openGaussAGEGraph instance."""

        self.graph_name = graph_name

        # check that psycopg2 is installed
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "Could not import psycopg2 python package. "
                "Please install it with `pip install psycopg2-binary`."
            )

        cfg = {
            "host": conf.host,
            "port": conf.port,
            "database": conf.database,
            "user": conf.user,
            "password": conf.password,
        }

        self.connection = psycopg2.connect(**cfg, **kwargs)

        with self._get_cursor() as curs:
            # check if graph with name graph_name exists
            graph_id_query = """SELECT namespaceoid \
                FROM ag_catalog.ag_graph \
                WHERE name = '{}'""".format(graph_name)

            curs.execute(graph_id_query)
            data = curs.fetchone()

            # if graph doesn't exist and create is True, create it
            if data is None:
                if create:
                    create_statement = """
                        SELECT ag_catalog.create_graph('{}');
                    """.format(graph_name)

                    try:
                        curs.execute(create_statement)
                        self.connection.commit()
                    except psycopg2.Error as e:
                        raise AGEQueryException(
                            {
                                "message": "Could not create the graph",
                                "detail": str(e),
                            }
                        )

                else:
                    raise Exception(
                        (
                            'Graph "{}" does not exist in the database '
                            + 'and "create" is set to False'
                        ).format(graph_name)
                    )

                curs.execute(graph_id_query)
                data = curs.fetchone()

            # store graph id
            self.graphid = data.namespaceoid

    @staticmethod
    def _wrap_query(query: str, graph_name: str) -> str:
        """
        Convert a Cyper query to an openGauss Age compatible Sql Query.
        Handles combined queries with UNION/EXCEPT operators
        Args:
            query (str) : A valid cypher query, can include UNION/EXCEPT operators
            graph_name (str) : The name of the graph to query
        Returns :
            str : An equivalent pgSql query wrapped with ag_catalog.cypher
        Raises:
            ValueError : If query is empty, contain RETURN *, or has invalid field names
        """

        if not query.strip():
            raise ValueError("Empty query provided")

        # pgsql template
        template = """SELECT {projection} FROM ag_catalog.cypher('{graph_name}', $$
            {query}
        $$) AS ({fields});"""

        # split the query into parts based on UNION and EXCEPT
        parts = re.split(r"\b(UNION\b|\bEXCEPT)\b", query, flags=re.IGNORECASE)

        all_fields = []

        for part in parts:
            if part.strip().upper() in ("UNION", "EXCEPT"):
                continue

            # if there are any returned fields they must be added to the pgsql query
            return_match = re.search(r'\breturn\b(?![^"]*")', part, re.IGNORECASE)
            if return_match:
                # Extract the part of the query after the RETURN keyword
                return_clause = part[return_match.end() :]

                # parse return statement to identify returned fields
                fields = (
                    return_clause.lower()
                    .split("distinct")[-1]
                    .split("order by")[0]
                    .split("skip")[0]
                    .split("limit")[0]
                    .split(",")
                )

                # raise exception if RETURN * is found as we can't resolve the fields
                clean_fileds = [f.strip() for f in fields if f.strip()]
                if "*" in clean_fileds:
                    raise ValueError(
                        "openGauss Age does not support RETURN * in Cypher queries"
                    )

                # Format fields and maintain order of appearance
                for idx, field in enumerate(clean_fileds):
                    field_name = AGEGraph._get_col_name(field, idx)
                    if field_name not in all_fields:
                        all_fields.append(field_name)

        # if no return statements found in any part
        if not all_fields:
            fields_str = "a agtype"

        else:
            all_fields = [field.replace(".", "_") for field in all_fields]
            fields_str = ", ".join(f"{field} agtype" for field in all_fields)

        return template.format(
            graph_name=graph_name,
            query=query,
            fields=fields_str,
            projection="*",
        )

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """
        insert a list of graph documents into the graph
        Args:
            graph_documents (List[GraphDocument]): the list of documents to be inserted
            include_source (bool): if True add nodes for the sources
                with MENTIONS edges to the entities they mention
        Returns:
            None
        """
        # query for inserting nodes
        node_insert_query = (
            """
            MERGE (n:`{label}` {{`id`: "{id}"}})
            set n.properties = [{properties}]
            """
            if not include_source
            else """
            MERGE (n:`{label}` {properties})
            MERGE (d:Document {d_properties})
            MERGE (d)-[:MENTIONS]->(n)
        """
        )

        # query for inserting edges
        edge_insert_query = """
            MERGE (from:`{f_label}` {f_properties})
            MERGE (to:`{t_label}` {t_properties})
            MERGE (from)-[:`{r_label}` {r_properties}]->(to)
        """
        # iterate docs and insert them
        for doc in graph_documents:
            # if we are adding sources, create an id for the source
            if include_source:
                if not doc.source.metadata.get("id"):
                    doc.source.metadata["id"] = md5(
                        doc.source.page_content.encode("utf-8")
                    ).hexdigest()

            # insert entity nodes
            for node in doc.nodes:
                node.properties["id"] = node.id
                if include_source:
                    query = node_insert_query.format(
                        label=node.type,
                        properties=self._format_properties(node.properties),
                        d_properties=self._format_properties(doc.source.metadata),
                    )
                else:
                    query = node_insert_query.format(
                        label=AGEGraph.clean_graph_labels(node.type),
                        properties=self._format_properties(node.properties),
                        id=node.id,
                    )

                self.query(query)

            # insert relationships
            for edge in doc.relationships:
                edge.source.properties["id"] = edge.source.id
                edge.target.properties["id"] = edge.target.id
                inputs = {
                    "f_label": AGEGraph.clean_graph_labels(edge.source.type),
                    "f_properties": self._format_properties(edge.source.properties),
                    "t_label": AGEGraph.clean_graph_labels(edge.target.type),
                    "t_properties": self._format_properties(edge.target.properties),
                    "r_label": AGEGraph.clean_graph_labels(edge.type).upper(),
                    "r_properties": self._format_properties(edge.properties),
                }

                query = edge_insert_query.format(**inputs)
                self.query(query)