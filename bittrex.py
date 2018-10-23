import requests
import json
import time
from datetime import datetime
from datastorage import DataCollector




def pull_data(pair):
    global unixtime_old

    url = 'https://bittrex.com/api/v1.1/public/getmarkethistory?market=' + pair

    try:
        r = requests.get(url)

        data = json.loads(r.text)
        price = data['result'][0]['Price']
        price_datetime = data['result'][0]['TimeStamp']

        # print('price: ', str(price))
        # print('price_datetime: ', str(price_datetime))

        return str(price), str(price_datetime)

    except Exception as e:
        print(e)
        print('request failed')



if __name__ == "__main__":

    parameters = []
    parameters.append({'name': 'btcprice', 'maxlength': 100})
    bittrexbtcprices = DataCollector(filename='bittrexbtcprices', parameters=parameters, overwrite=False,
                                  checklength=True)

    parameters = []
    parameters.append({'name': 'tetherprice', 'maxlength': 100})
    bittrextetherprices = DataCollector(filename='bittrextetherprices', parameters=parameters, overwrite=False,
                                     checklength=True)

    parameters = []
    parameters.append({'name': 'ethereumprice', 'maxlength': 100})
    ethereumprices = DataCollector(filename='ethereumprices', parameters=parameters, overwrite=False,
                                        checklength=True)

    parameters = []
    parameters.append({'name': 'bchprice', 'maxlength': 100})
    bchprices = DataCollector(filename='bchprices', parameters=parameters, overwrite=False,
                                   checklength=True)

    while True:

        try:
            # collect price every 5 minutes
            time.sleep(300)

            price, price_datetime = pull_data('USDT-BTC')
            print('price: ', price)
            print('appending bitcoin price at {}'.format(datetime.now()))
            bittrexbtcprices.append_data(parameter='btcprice', data=[price], times=[price_datetime])
            time.sleep(1)

            price, price_datetime = pull_data('USD-USDT')
            print('price: ', price)
            print('appending tether price at {}'.format(datetime.now()))
            bittrextetherprices.append_data(parameter='tetherprice', data=[price], times=[price_datetime])
            time.sleep(1)

            price, price_datetime = pull_data('USD-ETH')
            print('price: ', price)
            print('appending tether price at {}'.format(datetime.now()))
            ethereumprices.append_data(parameter='ethereumprice', data=[price], times=[price_datetime])
            time.sleep(1)

            price, price_datetime = pull_data('USD-BCH')
            print('price: ', price)
            print('appending tether price at {}'.format(datetime.now()))
            bchprices.append_data(parameter='bchprice', data=[price], times=[price_datetime])
            time.sleep(1)


        except Exception as e:
            print(e)




