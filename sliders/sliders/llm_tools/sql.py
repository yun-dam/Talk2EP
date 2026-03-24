import duckdb
import pandas as pd


class DuckSQLBasic:
    def __init__(self, db=":memory:"):
        self.conn = duckdb.connect(db)

    def __enter__(self):  # allow `with DuckSQLBasic() as db: …`
        return self

    def __exit__(self, exc_type, exc, tb):
        self.conn.close()

    def sql(self, query: str):
        return self.conn.sql(query)

    def register(self, df: pd.DataFrame, name: str):
        self.conn.register(name, df)

    def unregister(self, name: str):
        self.conn.unregister(name)

    def close(self):
        self.conn.close()


class DuckSQL:
    con = None

    def __init__(self, id_column="id", value_column="value"):
        if DuckSQL.con is None:
            DuckSQL.con = duckdb.connect()
        self.con = DuckSQL.con
        DuckSQL._row_id_column = id_column
        DuckSQL._row_value_column = value_column

        # Register UDFs that only look at our in-memory cache
        self.con.create_function(
            name="filter_by",
            function=DuckSQL.choose,
            parameters=[duckdb.typing.BIGINT, duckdb.typing.VARCHAR],
            return_type=duckdb.typing.BOOLEAN,
            side_effects=True,
        )
        self.con.create_function(
            name="cluster_on",
            function=DuckSQL.assign_cluster,
            parameters=[duckdb.typing.BIGINT, duckdb.typing.VARCHAR],
            return_type=duckdb.typing.VARCHAR,
            side_effects=True,
        )

    def register(self, df: pd.DataFrame, name: str):
        # Cache the full table once, then register for SQL
        DuckSQL._rows = df.to_dict("records")
        self.con.register(name, df)

    def sql(self, query: str):
        return self.con.sql(query)


def run_sql_query(query: str, duck_sql_conn: DuckSQLBasic):
    """Run a SQL query on the database."""
    try:
        return duck_sql_conn.sql(query).to_df().to_markdown(), False
    except Exception as e:
        # print(f"Error running SQL query: {e}")
        # print(f"Query: {query}")
        return f"Error running SQL query: {query}\n{e}. Please try again.", True
