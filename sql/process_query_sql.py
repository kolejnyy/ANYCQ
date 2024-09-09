import duckdb
import pandas as pd
from time import time
from src.data.query_to_sql import query_to_sql


kg_graph = pd.read_csv('data/FB15k-237-EFO1/test_kg.tsv', sep='\t', header = None)
cq_type = "(r2(e9,e7))&(r6(f1,e10))&(r9(f1,e2))&(r2(e9,s6))&(r10(e15,e5))&(r10(f1,e5))&(r13(e11,e2))&(r4(e9,e3))&(r1(s12,e14))&(r10(e14,e5))&(r2(e15,e7))&(r14(e11,e2))&(r3(e13,e10))&(r4(f1,e3))&(r12(e2,e11))&(r11(e14,s4))&(r6(e10,f1))&(r3(e7,f1))&(r8(e1,f1))&(r3(e7,e14))&(r11(f1,s4))&(r2(f1,e13))&(r3(e13,f1))&(r7(e14,e1))&(r5(e3,e15))"
grounding = {"r1": 13, "r2": 34, "r3": 35, "r4": 86, "r5": 87, "r6": 90, "r7": 96, "r8": 97, "r9": 120, "r10": 170, "r11": 200, "r12": 413, "r13": 438, "r14": 468, "s4": 160, "s6": 818, "s12": 4929}


kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

sql_query = query_to_sql(cq_type, grounding)
print(sql_query)

duckdb.sql("SET threads TO 16")
duckdb.sql("SET memory_limit TO '124GB'")

duckdb.sql("CREATE TABLE graph AS SELECT * FROM kg_graph")
duckdb.sql("INSERT INTO graph SELECT * FROM kg_graph")

s_t = time()
x = duckdb.sql(sql_query)
print(x)
print(time()-s_t)