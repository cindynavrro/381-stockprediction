import yfinance as yh
import matplotlib.pyplot as plt
import pandas as pd
from model2 import prediction_model


def get_data(t):
    return yh.Ticker(t).info


def get_market_data(t, period):
    return yh.Ticker(t).history(period=period)


def format_company_data(t):
    company_data = get_data(t)
    for key, value in company_data.items():
        print(key, ":", value)


# Outputs the time range options available for stocks
def output_time_options():
    time_options = {
        'A': 'ytd',
        'F': '6mo',
        'B': '10y',
        'G': '3mo',
        'C': '5y',
        'H': '1mo',
        'D': '2y',
        'I': '5d',
        'E': '1y',
        'J': '1d'
    }
    count = 0
    for key, value in time_options.items():
        if count == 1:
            count = 0
            print(key, ":", value)
        else:
            print(key, ":", value, end='\t\t')
            count += 1
    return time_options


# Extracts company data to cvs file
def write_to_cvs(t, p):
    df = get_market_data(t, p)
    df.to_csv(f'{t}.csv', header=True, index=None, sep=',', mode='a')
    return df


def get_graph(t):
    # df = get_market_data(t, '5y')
    # df.plot('Date', 'Close', color="red")
    # plt.show()
    df = pd.read_csv(f"{t}.csv")
    df.plot('date', 'close', color="red")
    plt.show()


def get_listing():
    df = pd.read_csv('nasdaq_listings.csv')
    listings = df['Symbol'].values.tolist()
    print("           Total Tickers:", len(listings), "for viewing")
    print("---------------------------------------------------------")
    print("Tickers will display 10 at a time. Press Enter to view...")
    print("...Or press N to exit")
    range_start = 0
    range_end = 10
    select = ''
    while select != 'N':
        select = str(input()).upper()
        if select != 'N':
            print(listings[range_start:range_end], end="")
            range_start = range_end
            range_end += 10


def print_welcome():
    print("-----------------------------------------------")
    print("|     Welcome to the Market Analytics v1.0     |")
    print("-----------------------------------------------")


def print_predict():
    print("-----------------------------------------------------------")
    print("Please wait while we retrieve most recent prediction models")
    print("**************** models based off 5yr data ****************")
    print("-----------------------------------------------------------")


def print_select1():
    print("A) Company Data")
    print("B) Market Data")
    select = str(input("Select A or B: ")).upper()
    return select


def print_select2():
    select = str(input("Would you like to view another stock? Y or N")).upper()
    return select

def print_select3(ticker):
    options = output_time_options()
    select = str(input("Enter the letter that corresponds with the time frame: ")).upper()
    time_frame = select
    print(get_market_data(ticker, options[select]))
    select = str(input("Would you like to export data to cvs? Y or N: ")).upper()
    if select == 'Y':
        write_to_cvs(ticker, time_frame)
        print("Data saved to CVS!")


def print_select4():
    select = str(input("Would you like to return to main? Y or N")).upper()
    if select == 'Y':
        print_select1()
        select = 'B'
    return select

def get_ticker():
    ticker = str(input("Enter Company Ticker Symbol: ")).upper()
    return ticker


def market_menu():
    ticker = get_ticker()
    select = print_select1()
    if select == 'A':
        format_company_data(ticker)
    select = print_select4()
    if select == 'B':
        print_select3(ticker)
        print_predict()
        prediction_model(company=ticker)
        select = print_select2()
        while select != 'N':
            if select != 'N':
                t = get_ticker()
                print_select3(t)
                prediction_model(company=t)
                select = print_select2()
                if select == 'N':
                    print("Goodbye.")


def main():
    print("Would you like to view some available ticker symbols for viewing?")
    select = str(input("Y or N: ")).upper()
    if select == 'Y':
        get_listing()
        market_menu()
    elif select == 'N':
        market_menu()
    else:
        print("Invalid option. Bye")


if __name__ == '__main__':
    print_welcome()
    main()
