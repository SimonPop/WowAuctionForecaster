import mysql.connector
import pandas as pd

def get_item_dataframe_by_month(item_id, slug = '"outland"', region = '"EU"'):
    """
    Makes a SQL request from the Undermine Journal to get the full DataFrame from an object.
    """

    mydb = mysql.connector.connect(
        host="newswire.theunderminejournal.com",
        user="",
        password="",
        database="newsstand"
    )

    mycursor = mydb.cursor()

    mycursor.execute("""select *
    from tblItemHistoryMonthly h
    join tblRealm r on h.house = r.house
    where r.region = {}
    and r.slug = {}
    and h.item = {}
    and h.level = {}
    order by h.month asc""".format(region, slug, item_id, 0))

    myresult = mycursor.fetchall()

    df = pd.DataFrame(myresult)
    df.columns = [line[0] for line in mycursor.description]
    return df

def get_item_timeseries(item_id, slug = '"outland"', region = '"EU"'):
    """
    Request the Undermine Journal DB for one item's timeseries.
    """
    mydb = mysql.connector.connect(
        host="newswire.theunderminejournal.com",
        user="",
        password="",
        database="newsstand"
    )

    mycursor = mydb.cursor()

    mycursor.execute("""select *
    from tblItemHistoryDaily h
    join tblRealm r on h.house = r.house
    where r.region = {}
    and r.slug = {}
    and h.item = {}
    order by h.when asc""".format(region, slug, item_id, 0))

    myresult = mycursor.fetchall()

    df = pd.DataFrame(myresult)
    df.columns = [line[0] for line in mycursor.description]

    return df

def get_all_df_from_graph(graph, slug = '"outland"', region = '"EU"'):
    from time import sleep 
    from tqdm import tqdm 
    import networkx as nx 
    """Queries all dataframes for all items in a graph."""
    item_ids = nx.get_node_attributes(graph,'item_id').values()
    item_dfs = {}
    for item_id in tqdm(item_ids):
        sleep(20)
        try:
            df = get_item_timeseries(item_id, slug = '"outland"', region = '"EU"')
            item_dfs[item_id] = df
        except:
            print("An exception occurred for {}".format(item_id))
    return item_dfs