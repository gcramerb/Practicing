import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"
def calculate_profit_n_loss(df:pd.DataFrame):
    filtered_df = df[df['action'] == "filled"]
    filtered_df.loc[:,"orderSide"] = filtered_df.loc[:,"orderSide"].map({'buy':-1, 'sell':1})
    filtered_df.loc[:,"operation_value"] = filtered_df.loc[:,"tradePx"] * filtered_df.loc[:,"tradeAmt"] * filtered_df.loc[:,"orderSide"]
    filtered_df.loc[:,"profit_n_loss"] = filtered_df.loc[:,"operation_value"].cumsum()
    return filtered_df


if __name__=="__main__":
    cumulative_PnL = []
    product_PnL = {}
    df= pd.read_csv("./Dev_test_task/profit_and_loss/data/test_logs.csv")
    df_prof_loss = calculate_profit_n_loss(df)
    df_by_product = [x for _, x in df.groupby('orderProduct')]
    for df_i in df_by_product:
        df_by_product_i = calculate_profit_n_loss(df_i)
        product_PnL[df_by_product_i.iloc[0]["orderProduct"]] = df_by_product_i.iloc[-1]["profit_n_loss"]    
    totalPnL = df_prof_loss.iloc[-1]["profit_n_loss"]
    print(f"Total gross PnL: {totalPnL}")
    print(f"Total gross PnL over each security ID: {product_PnL}")


    df_plot = pd.melt(df_prof_loss, id_vars='currentTime', value_vars='profit_n_loss')
    fig = px.line(df_plot, x='currentTime', y='value')
    min_x = df_prof_loss.iloc[0]['currentTime']
    max_x = df_prof_loss.iloc[-1]['currentTime']
    fig.update_xaxes(range = [min_x,max_x])
    fig.show()
    fig.write_image("./Dev_test_task/profit_and_loss/output/profit_n_loss.png")