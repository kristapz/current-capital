import os
import json
from openai import OpenAI
from google.cloud import bigquery
from google.cloud import aiplatform
import time
from typing import List

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey/test1-key.json'
project_id = 'test1-427219'
dataset_id = 'indicators'
table_id = 'fred_indicator'
full_table_id = f"{project_id}.{dataset_id}.{table_id}"

# Initialize BigQuery Client and Vertex AI
client = bigquery.Client(project=project_id)
aiplatform.init(project=project_id, location='us-east1')

# Vertex AI Endpoint
vertex_endpoint_name = "projects/510158460879/locations/us-east1/endpoints/7398750082746548224"
vertex_endpoint = aiplatform.Endpoint(endpoint_name=vertex_endpoint_name)

# Initialize OpenAI client
openai_client = OpenAI(
    api_key="sk-None-WB2bX4Z2lFWuobMohrUuT3BlbkFJZZsmV3aoiqnN1leZJxW9")  # Replace with your actual API key


def generate_vertex_embeddings(text):
    text = str(text).strip()
    if text == '' or text.lower() == 'nan':
        print("Empty or NaN text encountered, returning empty embedding for Vertex AI.")
        return []

    max_characters = 1450
    cleaned_text = ' '.join(text[:max_characters].split())
    instances = [{"inputs": cleaned_text}]
    try:
        response = vertex_endpoint.predict(instances=instances)
        return response.predictions[0]
    except Exception as e:
        print(f"Error generating Vertex AI embeddings: {e}")
        return []


def generate_openai_embeddings(text):
    text = str(text).strip()
    if text == '' or text.lower() == 'nan':
        print("Empty or NaN text encountered, returning empty embedding for OpenAI.")
        return []

    max_characters = 8000  # text-embedding-3-large can handle up to 8191 tokens
    cleaned_text = ' '.join(text[:max_characters].split())
    try:
        response = openai_client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating OpenAI embeddings: {e}")
        return []


def insert_indicator_with_embeddings(indicator: dict):
    ticker = indicator['ticker']
    description = indicator['long_business_summary']

    print(f"Processing Indicator: {ticker}")

    vertex_embeddings = generate_vertex_embeddings(description)
    openai_embeddings = generate_openai_embeddings(description)

    if not vertex_embeddings or not openai_embeddings:
        print(f"Skipping insertion for {ticker} due to missing embeddings.")
        return

    insert_query = f"""
    INSERT INTO `{full_table_id}` (Indicator, Name, Description, Embedding_Model1, Embedding_Model2)
    VALUES (
        '{ticker}',
        '{sector_metrics_inv.get(ticker, "No Name Available")}',
        '{description.replace("'", "''")}',
        '{json.dumps(vertex_embeddings)}',
        '{json.dumps(openai_embeddings)}'
    )
    """

    try:
        query_job = client.query(insert_query)
        query_job.result()
        print(f"Successfully inserted Indicator: {ticker}")
    except Exception as e:
        print(f"Exception occurred while inserting Indicator: {ticker}: {e}")


def insert_all_indicators(indicators: List[dict]):
    for indicator in indicators:
        insert_indicator_with_embeddings(indicator)
        time.sleep(1)  # Optional: Pause to respect API rate limits


# Updated sector_metrics with new indicators
sector_metrics = {
    'Overall Economic Indicators': {
        'GDP': 'Gross Domestic Product (GDP)',
        'GDPC1': 'Real Gross Domestic Product',
        'A939RC0Q052SBEA': 'Gross Domestic Product Per Capita',
        'M2SL': 'M2 Money Stock',
        'M2V': 'Velocity of M2 Money Stock',
        'GFDEBTN': 'Total Public Debt',
        'GFDEGDQ188S': 'Federal Debt: Total Public Debt as Percent of GDP',
        'BOPGSTB': 'U.S. Trade Balance',
        'IEABC': 'Current Account Balance',
        'CP': 'Corporate Profits After Tax',
        'NETEXP': 'Net Exports of Goods and Services',
    },
    'Labor Market Indicators': {
        'UNRATE': 'Unemployment Rate',
        'CIVPART': 'Civilian Labor Force Participation Rate',
        'PAYEMS': 'Total Nonfarm Payrolls',
        'CES0500000003': 'Average Hourly Earnings of All Employees: Total Private',
        'AWHAETP': 'Average Weekly Hours of Production and Nonsupervisory Employees: Total Private',
        'JTSJOL': 'Job Openings: Total Nonfarm',
        'JTSQUR': 'Quits: Total Nonfarm',
        'ICSA': 'Initial Claims',
        'CCSA': 'Continuing Claims',
        'DRTSCIS': 'Senior Loan Officer Opinion Survey on Bank Lending Practices',
    },
    'Inflation Indicators': {
        'CPIAUCSL': 'Consumer Price Index for All Urban Consumers: All Items',
        'CPILFESL': 'Consumer Price Index for All Urban Consumers: All Items Less Food and Energy (Core CPI)',
        'PPIACO': 'Producer Price Index by Commodity: All Commodities',
        'PCE': 'Personal Consumption Expenditures (PCE)',
        'CPIMEDSL': 'Consumer Price Index: Medical Care',
        'CPIENGSL': 'Consumer Price Index: Energy',
        'CPIFABSL': 'Consumer Price Index: Food',
        'PCEC96': 'Real Personal Consumption Expenditures',
    },
    'Housing Market Indicators': {
        'HOUST': 'Housing Starts: Total New Privately Owned',
        'PERMIT': 'Building Permits: New Private Housing Units',
        'EXHOSLUSM495S': 'Existing Home Sales',
        'HSN1F': 'New Residential Sales',
        'CSUSHPINSA': 'S&P/Case-Shiller U.S. National Home Price Index',
        'MORTGAGE30US': '30-Year Fixed Rate Mortgage Average',
    },
    'Consumer Indicators': {
        'RSAFS': 'Retail Sales: Total',
        'UMCSENT': 'University of Michigan: Consumer Sentiment',
        'CSCICP03USM665S': 'Consumer Confidence Index',
        'PSAVERT': 'Personal Saving Rate',
        'DSPIC96': 'Real Disposable Personal Income',
    },
    'Industrial Indicators': {
        'INDPRO': 'Industrial Production Index',
        'TCU': 'Capacity Utilization: Total Industry',
        'BUSINV': 'Business Inventories',
        'ISRATIO': 'Inventory to Sales Ratio',
        'DGORDER': 'Manufacturers\' New Orders: Durable Goods',
        'CMRMTSPL': 'Total Business Sales',
        'GPDI': 'Gross Private Domestic Investment',
    },
    'Financial Market Indicators': {
        'SP500': 'S&P 500',
        'DJIA': 'Dow Jones Industrial Average',
        'NASDAQCOM': 'NASDAQ Composite Index',
        'VIXCLS': 'CBOE Volatility Index: VIX',
        'DGS10': '10-Year Treasury Constant Maturity Rate',
        'FEDFUNDS': 'Effective Federal Funds Rate',
        'BAMLH0A0HYM2': 'High Yield Corporate Bond Spread',
        'DPRIME': 'Bank Prime Loan Rate',
        'DAAA': 'Corporate Bond Yield Rates (AAA)',
        'DBAA': 'Corporate Bond Yield Rates (BAA)',
        'TEDRATE': 'TED Spread',
        'T10Y2Y': 'Interest Rate Spreads: 10-Year Treasury vs. 2-Year Treasury',
        'NFCI': 'National Financial Conditions Index',
        'STLFSI2': 'St. Louis Fed Financial Stress Index',
    },
    'Energy Market Indicators': {
        'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate (WTI)',
        'MHHNGSP': 'Natural Gas Prices',
    },
    # Retaining existing sectors
    'Technology Sector': {
        'NASDAQCOM': 'NASDAQ Composite Index',
        'CEU5000000001': 'All Employees: Information',
    },
    'Healthcare Industry': {
        'CEU6562000001': 'All Employees: Health Care and Social Assistance',
        'CPIMEDSL': 'Consumer Price Index: Medical Care',
    },
    'Financial Services': {
        'CEU5500000001': 'All Employees: Financial Activities',
        'FEDFUNDS': 'Effective Federal Funds Rate',
        'TOTLL': 'Total Loans and Leases, All Commercial Banks',
        'DPRIME': 'Bank Prime Loan Rate',
        'BUSLOANS': 'Commercial and Industrial Loans at All Commercial Banks',
        'BAMLH0A0HYM2': 'High Yield Corporate Bond Spread',
    },
    'Energy Sector (Oil & Gas)': {
        'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate (WTI)',
        'DHHNGSP': 'Henry Hub Natural Gas Spot Price',
        'CEU1021000001': 'All Employees: Mining',
        'IPMINE': 'Industrial Production: Mining',
    },
    'Utilities Industry': {
        'IPUTIL': 'Industrial Production: Utilities',
        'CEU4422000001': 'All Employees: Utilities',
        'CUSR0000SEHF02': 'CPI for All Urban Consumers: Utility (Piped) Gas Service',
    },
    'Consumer Staples': {
        'PCND': 'Personal Consumption Expenditures: Nondurable Goods',
        'CPIUFDSL': 'Consumer Price Index for All Urban Consumers: Food and Beverages',
        'CEU3100000001': 'All Employees: Food Manufacturing',
    },
    'Consumer Discretionary': {
        'RSXFS': 'Advance Retail Sales: Retail Trade',
        'CEU4200000001': 'All Employees: Retail Trade',
        'UMCSENT': 'Consumer Sentiment Index',
    },
    'Industrial Goods': {
        'DGORDER': 'Durable Goods Orders',
        'IPMAN': 'Industrial Production: Manufacturing',
        'CEU3000000001': 'All Employees: Manufacturing',
    },
    'Real Estate': {
        'CSUSHPINSA': 'S&P/Case-Shiller U.S. National Home Price Index',
        'HOUST': 'Housing Starts',
        'CEU2000000001': 'All Employees: Construction',
    },
    'Transportation': {
        'CEU4300000001': 'All Employees: Transportation and Warehousing',
        'IPG336S': 'Industrial Production: Durable Goods: Transportation Equipment',
        'TSIFRGHT': 'Transportation Services Index: Freight',
    },
    'Mining and Metals': {
        'PCU212230212230': 'PPI: Copper, Nickel, Lead, and Zinc Mining',
        'CEU1021100001': 'All Employees: Metal Ore Mining',
    },
    'Agriculture': {
        'CEU0500000001': 'All Employees: Total Private',
    },
    'Retail Industry': {
        'RSAFS': 'Advance Retail Sales: Retail and Food Services, Total',
        'RETAILIRSA': 'Retailers Inventories to Sales Ratio',
        'CEU4200000001': 'All Employees: Retail Trade',
    },
    'Education and Health Services': {
        'CEU6500000001': 'All Employees: Education and Health Services',
        'CPIEDUSL': 'Consumer Price Index for All Urban Consumers: Education',
    },
    'Hospitality and Leisure': {
        'CEU7000000001': 'All Employees: Leisure and Hospitality',
        'CPIRECNS': 'Consumer Price Index for All Urban Consumers: Recreation',
    },
    'Construction': {
        'CEU2000000001': 'All Employees: Construction',
        'TTLCONS': 'Construction Spending: Total',
        'PERMIT': 'New Private Housing Authorized by Building Permits',
    },
    'Automotive Industry': {
        'TOTALSA': 'Total Vehicle Sales',
        'IPG3361T3S': 'Industrial Production: Motor Vehicles and Parts',
    },
    'Information Technology Services': {
        'CEU6000000001': 'All Employees: Professional and Business Services',
        'PCU511210511210': 'PPI by Industry: Software Publishers',
        'IPG334S': 'Industrial Production: Computer and Electronic Product',
    },
    'Chemical Manufacturing': {
        'IPG325S': 'Industrial Production: Nondurable Goods: Chemical',
    },
}

# Updated indicator_descriptions with new indicators
indicator_descriptions = {
    'GDP': 'Gross Domestic Product (GDP) represents the total monetary value of all goods and services produced over a specific time period within the United States, serving as a comprehensive scorecard of the country’s economic health.',
    'GDPC1': 'Real Gross Domestic Product (Real GDP) adjusts the nominal GDP by removing the effects of inflation, providing a more accurate reflection of an economy’s size and how it’s growing over time in constant dollars.',
    'A939RC0Q052SBEA': 'Gross Domestic Product Per Capita divides the total GDP by the number of people in the country, offering a per-person measure of economic output and a general gauge of prosperity and living standards.',
    'UNRATE': 'Unemployment Rate measures the percentage of the total labor force that is unemployed but actively seeking employment and willing to work, indicating the overall slack in the labor market.',
    'CIVPART': 'Civilian Labor Force Participation Rate reflects the proportion of the working-age population that is either employed or actively looking for work, highlighting the active portion of the economy’s labor pool.',
    'PAYEMS': 'Total Nonfarm Payrolls encompass the total number of paid U.S. workers of any business, excluding general government employees, private household employees, employees of nonprofit organizations, and farm employees.',
    'CPIAUCSL': 'Consumer Price Index for All Urban Consumers (CPI-U): All Items tracks changes in the price level of a market basket of consumer goods and services purchased by urban households, serving as a key indicator of inflation.',
    'CPILFESL': 'Consumer Price Index for All Urban Consumers: All Items Less Food and Energy (Core CPI) excludes volatile food and energy prices to provide a clearer view of the underlying, persistent trends in inflation.',
    'PPIACO': 'Producer Price Index by Commodity: All Commodities measures the average change over time in the selling prices received by domestic producers for their output, reflecting inflation at the wholesale level.',
    'PCE': 'Personal Consumption Expenditures (PCE) represent the value of goods and services purchased by or on behalf of U.S. residents, providing insight into consumer spending patterns and economic well-being.',
    'RSAFS': 'Retail Sales: Total measure the total receipts at retail stores, indicating consumer spending trends and serving as a leading indicator of the economy’s health.',
    'INDPRO': 'Industrial Production Index measures the real output of all relevant establishments located in the United States manufacturing, mining, and electric, and gas utilities, signaling industrial sector health.',
    'TCU': 'Capacity Utilization: Total Industry indicates the percentage of resources used by corporations and factories to produce goods in the U.S., reflecting overall economic efficiency and potential inflationary pressure.',
    'HOUST': 'Housing Starts: Total New Privately Owned tracks the number of new residential construction projects begun during a particular month, serving as a key indicator of economic strength.',
    'PERMIT': 'Building Permits: New Private Housing Units represent the authorization for new housing units, forecasting future construction activity and housing supply.',
    'EXHOSLUSM495S': 'Existing Home Sales reflect the number of previously constructed homes that were sold during the month, providing insight into housing demand and consumer confidence.',
    'HSN1F': 'New Residential Sales capture the sales of newly built single-family homes, indicating economic momentum in the housing sector and broader economy.',
    'CSUSHPINSA': 'S&P/Case-Shiller U.S. National Home Price Index measures the change in value of the U.S. residential housing market, offering a gauge of price trends and housing market health.',
    'MORTGAGE30US': '30-Year Fixed Rate Mortgage Average reports the average interest rate for a 30-year fixed-rate mortgage, affecting housing affordability and consumer borrowing costs.',
    'DGS10': '10-Year Treasury Constant Maturity Rate reflects the yield received for investing in a U.S. government-issued treasury security that has a maturity of ten years, serving as a benchmark for other interest rates.',
    'FEDFUNDS': 'Effective Federal Funds Rate is the interest rate at which depository institutions trade federal funds with each other overnight, influencing monetary policy and economic activity.',
    'SP500': 'S&P 500 is a stock market index tracking the performance of 500 large companies listed on stock exchanges in the United States, representing the stock market’s performance and investor sentiment.',
    'DJIA': 'Dow Jones Industrial Average (DJIA) is a price-weighted index that tracks 30 large, publicly-owned blue-chip companies trading on the New York Stock Exchange (NYSE) and the NASDAQ, reflecting market trends.',
    'NASDAQCOM': 'NASDAQ Composite Index represents the performance of over 3,000 companies listed on the NASDAQ exchange.',
    'VIXCLS': 'CBOE Volatility Index: VIX estimates the expected volatility of the S&P 500 index over the next 30 days, often referred to as the "fear gauge" of the stock market.',
    'TOTALSA': 'Total Vehicle Sales represent the number of domestically produced cars and light-duty trucks that are sold, indicating consumer confidence and spending in the durable goods sector.',
    'UMCSENT': 'University of Michigan: Consumer Sentiment Index measures consumer confidence in economic activity, providing insights into consumer spending and saving behaviors.',
    'CSCICP03USM665S': 'Consumer Confidence Index reflects prevailing business conditions and likely developments for the months ahead, based on consumer attitudes, buying intentions, and overall economic expectations.',
    'CFNAI': 'Chicago Fed National Activity Index is a monthly index designed to gauge overall economic activity and related inflationary pressure, derived from 85 individual indicators.',
    'GFDEBTN': 'Total Public Debt encompasses all the money owed by the federal government, indicating the fiscal health and borrowing needs of the country.',
    'GFDEGDQ188S': 'Federal Debt: Total Public Debt as Percent of GDP provides a measure of the national debt in relation to the size of the economy, highlighting potential fiscal sustainability issues.',
    'BOPGSTB': 'U.S. Trade Balance shows the difference between the value of the country’s exports and imports, indicating the competitiveness of the U.S. economy and influencing currency values.',
    'IEABC': 'Current Account Balance records a nation’s transactions with the rest of the world—specifically its net trade in goods and services, net earnings on cross-border investments, and net transfer payments.',
    'M2SL': 'M2 Money Stock includes M1 plus savings deposits, money market securities, and other time deposits, indicating the money supply and potential inflationary pressures.',
    'M2V': 'Velocity of M2 Money Stock measures the frequency at which one unit of currency is used to purchase domestically-produced goods and services within a given time period, reflecting economic activity levels.',
    'RESBALNS': 'Total Reserves of Depository Institutions include the vault cash and deposits held by depository institutions at Federal Reserve Banks, indicating liquidity in the banking system.',
    'TOTALSL': 'Total Consumer Credit Owned and Securitized, Outstanding represents the total amount of credit extended to individuals for household, family, and other personal expenditures, excluding loans secured by real estate.',
    'DPRIME': 'Bank Prime Loan Rate is the rate that banks charge their most creditworthy customers, often serving as a benchmark for other loans and credit products.',
    'BUSINV': 'Business Inventories measure the dollar amount of inventories held by manufacturers, wholesalers, and retailers, indicating future production needs and economic momentum.',
    'ISRATIO': 'Inventory to Sales Ratio shows the relationship between the amount of inventory and the amount of sales, providing insights into supply chain efficiency and potential overstocking.',
    'CES0500000003': 'Average Hourly Earnings of All Employees: Total Private tracks the average hourly earnings, reflecting wage growth, consumer purchasing power, and potential inflationary pressures.',
    'AWHAETP': 'Average Weekly Hours of Production and Nonsupervisory Employees: Total Private measures the average number of hours worked, indicating labor market utilization and potential output capacity.',
    'JTSJOL': 'Job Openings: Total Nonfarm represents the number of available job positions that are unfilled, signaling labor demand and economic health.',
    'JTSQUR': 'Quits: Total Nonfarm tracks the number of employees who voluntarily leave their jobs, indicating worker confidence in finding new employment and labor market fluidity.',
    'ICSA': 'Initial Claims measure the number of individuals filing for unemployment benefits for the first time, providing a timely indicator of labor market conditions.',
    'CCSA': 'Continuing Claims represent the number of individuals receiving unemployment benefits after their initial claim, indicating longer-term unemployment trends.',
    'PSAVERT': 'Personal Saving Rate shows the percentage of disposable income that households save rather than spend, reflecting consumer confidence and future spending capacity.',
    'CP': 'Corporate Profits After Tax indicate the profitability of corporations after accounting for taxes, influencing investment, stock valuations, and economic growth.',
    'BAMLH0A0HYM2': 'High Yield Corporate Bond Spread measures the difference in yields between high-yield bonds and comparable maturity Treasury bonds, reflecting investor risk appetite and credit market conditions.',
    'CPIMEDSL': 'Consumer Price Index: Medical Care tracks changes in the price level of medical goods and services, affecting healthcare affordability and overall inflation.',
    'CPIENGSL': 'Consumer Price Index: Energy measures changes in the cost of energy commodities and services, influencing consumer expenses and business costs.',
    'CPIFABSL': 'Consumer Price Index: Food monitors changes in the price of food items, impacting household budgets and inflation rates.',
    'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate (WTI) reflects the spot price per barrel of West Texas Intermediate oil, influencing energy costs and economic conditions.',
    'MHHNGSP': 'Natural Gas Prices represent the spot price of natural gas delivered at the Henry Hub in Louisiana, affecting energy markets and consumer utility costs.',
    'TTLCONS': 'Total Construction Spending encompasses the total value of construction work done in the U.S., indicating economic activity in the construction sector.',
    'DSPIC96': 'Real Disposable Personal Income is the income remaining after deduction of taxes and adjusted for inflation, available to households for spending or saving.',
    'TWEXB': 'Trade Weighted U.S. Dollar Index: Broad measures the value of the U.S. dollar relative to the currencies of its major trading partners, influencing import/export prices and economic competitiveness.',
    'DEXUSEU': 'Exchange Rate: U.S. Dollar to Euro indicates how many U.S. dollars are needed to purchase one euro, affecting trade and investment between the U.S. and Eurozone.',
    'DEXJPUS': 'Exchange Rate: U.S. Dollar to Japanese Yen shows the value of the U.S. dollar in terms of Japanese yen, impacting trade balances and financial flows.',
    'DEXUSUK': 'Exchange Rate: U.S. Dollar to British Pound represents the exchange rate between the U.S. dollar and the British pound sterling, influencing bilateral trade and investment.',
    'DGORDER': 'Manufacturers\' New Orders: Durable Goods tracks the value of new orders placed with manufacturers for immediate and future delivery of durable goods, signaling manufacturing sector health.',
    'CMRMTSPL': 'Total Business Sales measure the aggregate sales of wholesale and retail businesses, indicating consumer demand and economic activity.',
    'MTSDS133FMS': 'Federal Surplus or Deficit reports the difference between federal government receipts and expenditures, reflecting fiscal policy and government borrowing needs.',
    'FGRECPT': 'Federal Government Current Receipts and Expenditures detail the federal government’s income and spending, providing insight into fiscal balance and economic impact.',
    'DCPF3M': 'Commercial Paper Rates reflect the average interest rates on unsecured, short-term debt instruments issued by corporations, indicating credit market conditions.',
    'T10Y2Y': 'Interest Rate Spreads: 10-Year Treasury vs. 2-Year Treasury measures the difference between long-term and short-term Treasury yields, often used as an indicator of economic expectations and potential recession.',
    'DRTSCIS': 'Senior Loan Officer Opinion Survey on Bank Lending Practices gathers information on the supply and demand for bank credit, indicating lending conditions and credit availability.',
    'NFCI': 'National Financial Conditions Index provides a comprehensive weekly update on U.S. financial conditions in money markets, debt and equity markets, and the traditional and "shadow" banking systems.',
    'T5YIE': '5-Year Breakeven Inflation Rate is derived from the difference between the yield on a nominal 5-year Treasury and a 5-year Treasury Inflation-Protected Security (TIPS), indicating market inflation expectations.',
    'T10YIE': '10-Year Breakeven Inflation Rate represents the expected rate of inflation over the next ten years, derived from the difference between nominal and inflation-protected Treasury securities.',
    'GPDI': 'Gross Private Domestic Investment measures the amount of money that domestic businesses invest within their own country, indicating confidence in future economic growth.',
    'PCEC96': 'Real Personal Consumption Expenditures adjust personal consumption expenditures for inflation, reflecting the real value of consumer spending.',
    'EXPGS': 'Exports of Goods and Services represent the value of all goods and other market services provided to the rest of the world, indicating international demand for domestic products.',
    'IMPGS': 'Imports of Goods and Services denote the value of all goods and other market services received from the rest of the world, indicating domestic demand for foreign products.',
    'VXVCLS': 'CBOE S&P 500 3-Month Volatility Index estimates expected volatility over a three-month period for the S&P 500 Index, serving as a barometer for market uncertainty.',
    'STLFSI2': 'St. Louis Fed Financial Stress Index measures the degree of financial stress in the markets, with a value of zero indicating normal financial market conditions.',
    'OUTMS': 'Manufacturing Sector: Real Output provides an inflation-adjusted measure of the manufacturing sector’s production, indicating industrial health and economic vitality.',
    'ECIALLCIV': 'Employment Cost Index: Total Compensation tracks changes in the costs of labor for businesses in the U.S. economy, including wages, salaries, and benefits, influencing inflation and monetary policy.',
    'DRCCLACBS': 'Delinquency Rate on Credit Card Loans, All Commercial Banks shows the percentage of credit card loans that are past due, indicating consumer financial stress and credit risk.',
    'DRALACBS': 'Delinquency Rate on Commercial and Industrial Loans, All Commercial Banks measures the percentage of commercial loans that are past due, reflecting business financial health and lending conditions.',
    'PI': 'Personal Income represents the total income received by individuals from all sources, including wages, investments, and government transfers, influencing consumer spending potential.',
    'MEHOINUSA672N': 'Real Median Household Income shows the median household income adjusted for inflation, providing insight into the economic well-being of the median household.',
    'QUSPAM770A': 'Total Credit to Private Non-Financial Sector indicates the total amount of credit extended to private non-financial businesses and households, signaling leverage and financial stability risks.',
    'DRCLACBS': 'Consumer Loan Delinquency Rate reflects the percentage of consumer loans (excluding real estate) that are past due, indicating consumer credit health and potential banking sector risks.',
    'FINSLC96': 'Real Final Sales of Domestic Product represents GDP minus the change in private inventories, offering a measure of the economy’s output excluding inventory fluctuations.',
    'NETEXP': 'Net Exports of Goods and Services calculate the value of a country’s total exports minus its total imports, influencing GDP and economic growth.',
    'DAAA': 'Corporate Bond Yield Rates (AAA) reflect the average yield on AAA-rated corporate bonds, indicating the cost of borrowing for the most creditworthy companies.',
    'DBAA': 'Corporate Bond Yield Rates (BAA) represent the average yield on BAA-rated corporate bonds, providing insight into borrowing costs for lower investment-grade companies.',
    'TEDRATE': 'TED Spread measures the difference between the 3-month LIBOR and the 3-month Treasury bill rate, indicating perceived credit risk in the general economy.',
    'TOTBKCR': 'Total Credit Market Assets Held by All Commercial Banks include all loans and securities held by banks, providing insight into bank lending activity and financial system health.',
    # Retaining existing indicator descriptions
    'CEU5000000001': 'Total number of employees working in the information sector across the U.S. economy, including IT, media, and telecommunications.',
    'CEU6562000001': 'Total number of employees in health care and social assistance sectors, covering hospitals, clinics, and social services.',
    'CEU5500000001': 'Total number of employees in the financial activities sector, including banking, insurance, and investment services.',
    'TOTLL': 'Total value of loans and leases held by all commercial banks in the U.S.',
    'BUSLOANS': 'Total value of loans and leases made to businesses by all commercial banks.',
    'DHHNGSP': 'The price of natural gas at the Henry Hub, a key U.S. distribution hub.',
    'CEU1021000001': 'Total number of employees in the mining sector, including extraction of minerals and resources.',
    'IPMINE': 'Measures the output of the mining sector, including extraction and processing of minerals.',
    'IPUTIL': 'Measures the output of the utilities sector, including electricity, gas, and water services.',
    'CEU4422000001': 'Total number of employees in the utilities sector, including electricity, gas, and water services.',
    'CUSR0000SEHF02': 'Consumer Price Index measuring the cost changes for piped gas service in urban areas.',
    'PCND': 'Measures the expenditures on nondurable goods consumed by households, such as food and clothing.',
    'CPIUFDSL': 'Measures changes in the cost of food and beverages purchased by urban consumers.',
    'CEU3100000001': 'Total number of employees in the food manufacturing sector, including processing and packaging.',
    'RSXFS': 'Monthly data on the total retail sales in the U.S., including various retail sectors.',
    'CEU4200000001': 'Total number of employees working in the retail trade sector across the U.S.',
    'UMCSENT': 'A measure of consumer confidence and their expectations about the economy\'s future performance.',
    'IPMAN': 'Measures the output of the manufacturing sector, including production of goods and materials.',
    'CEU3000000001': 'Total number of employees in the manufacturing sector across the U.S. economy.',
    'CEU2000000001': 'Total number of employees working in the construction sector, including residential and commercial.',
    'CEU4300000001': 'Total number of employees in the transportation and warehousing sector.',
    'IPG336S': 'Measures the output of durable transportation equipment, including vehicles and machinery.',
    'TSIFRGHT': 'An index measuring the performance of freight transportation services in the economy.',
    'PCU212230212230': 'Producer Price Index for mining of copper, nickel, lead, and zinc.',
    'CEU1021100001': 'Total number of employees in metal ore mining sectors, including extraction and processing.',
    'CEU0500000001': 'Total number of private sector employees across all industries in the U.S.',
    'RETAILIRSA': 'A measure of inventory levels held by retailers relative to their sales.',
    'CEU6500000001': 'Total number of employees in the education and health services sectors.',
    'CPIEDUSL': 'Measures changes in the cost of educational services for urban consumers.',
    'CEU7000000001': 'Total number of employees in leisure and hospitality sectors, including hotels and restaurants.',
    'CPIRECNS': 'Measures changes in the cost of recreational goods and services for urban consumers.',
    'PERMIT': 'Number of new housing units authorized by building permits issued.',
    'IPG3361T3S': 'Measures the output of motor vehicle and parts manufacturing sectors.',
    'CEU6000000001': 'Total number of employees in professional and business services sectors.',
    'PCU511210511210': 'Producer Price Index for the software publishing industry.',
    'IPG334S': 'Measures the output of computer and electronic product manufacturing sectors.',
    'IPG325S': 'Measures the output of chemical nondurable goods manufacturing sectors.',
}

# Create reverse mapping
sector_metrics_inv = {}
for sector, indicators in sector_metrics.items():
    for ticker, name in indicators.items():
        sector_metrics_inv[ticker] = name

# Prepare indicators data
indicators_data = []
for sector, indicators in sector_metrics.items():
    for ticker, name in indicators.items():
        description = indicator_descriptions.get(ticker, "No description available.")
        indicators_data.append({
            "ticker": ticker,
            "long_business_summary": description
        })

if __name__ == "__main__":
    # Check if the dataset exists; if not, create it
    try:
        client.get_dataset(dataset_id)
        print(f"Dataset '{dataset_id}' already exists.")
    except Exception as e:
        dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
        dataset.location = "US"
        dataset = client.create_dataset(dataset)
        print(f"Dataset '{dataset_id}' created.")

    # Define the schema for the new table
    schema = [
        bigquery.SchemaField("Indicator", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("Name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("Description", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("Embedding_Model1", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("Embedding_Model2", "STRING", mode="REQUIRED"),
    ]

    # Check if the new table exists; if not, create it
    table_ref = client.dataset(dataset_id).table(table_id)
    try:
        client.get_table(table_ref)
        print(f"Table '{full_table_id}' already exists.")
    except Exception as e:
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
        print(f"Table '{full_table_id}' created.")

    insert_all_indicators(indicators_data)
