import pandas as pd
import plotly.graph_objects as go

def calculate_profit_n_loss(df:pd.DataFrame):
    filtered_df = df[df['action'] == "filled"]
    filtered_df.loc[:,"orderSide"] = filtered_df["orderSide"].map({'buy':-1.0, 'sell':1.0})
    filtered_df = filtered_df.astype({'orderSide': 'float64'})
    filtered_df.loc[:,"operation_value"] = filtered_df.loc[:,"tradePx"] * filtered_df.loc[:,"tradeAmt"] * filtered_df.loc[:,"orderSide"]
    filtered_df.loc[:,"profit_n_loss"] = filtered_df.loc[:,"operation_value"].cumsum()
    return filtered_df

def plot_prof_n_loss(df):
    cutoff = 0
    x = df["currentTime"]
    y = df["profit_n_loss"]
    colors=['red' if val < cutoff else 'green' for val in y]
    trace = go.Scatter(
        x=x, 
        y=y, 
        mode='markers+lines', 
        marker={'color': colors}, 
        line={'color': 'gray'}
    )
    fig = go.Figure(data=trace)
    #fig.update_layout(title_text="Profit and Loss")
    fig.write_image("./Dev_test_task/profit_and_loss/output/profit_n_loss.pdf")
    fig.show()
if __name__=="__main__":
    cumulative_PnL = []
    product_PnL = {}
    df= pd.read_csv("./Dev_test_task/profit_and_loss/data/test_logs.csv", sep = ";")
    df_prof_loss = calculate_profit_n_loss(df)
    df_by_product = [x for _, x in df.groupby('orderProduct')]
    for df_i in df_by_product:
        df_by_product_i = calculate_profit_n_loss(df_i)
        product_PnL[df_by_product_i.iloc[0]["orderProduct"]] = df_by_product_i.iloc[-1]["profit_n_loss"]    
    totalPnL = df_prof_loss.iloc[-1]["profit_n_loss"]
    print(f"Total gross PnL: {totalPnL}")
    print(f"Total gross PnL over each security ID: {product_PnL}")
    plot_prof_n_loss(df_prof_loss)