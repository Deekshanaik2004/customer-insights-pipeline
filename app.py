import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
from sqlalchemy import create_engine

# Connect to PostgreSQL
engine = create_engine("postgresql://your_username:your_password@localhost:5432/customer_insights")
df = pd.read_sql("SELECT * FROM transactions", engine)

# Create Dash app
app = dash.Dash(__name__)

# Example Plot: Monthly Spending
df['invoice_date'] = pd.to_datetime(df['invoice_date'])
df['month'] = df['invoice_date'].dt.to_period('M').astype(str)
monthly_spending = df.groupby('month')['amount'].sum().reset_index()

fig = px.line(monthly_spending, x='month', y='amount', title='Monthly Total Spending')

# Layout
app.layout = html.Div([
    html.H1("📊 Customer Insights Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
