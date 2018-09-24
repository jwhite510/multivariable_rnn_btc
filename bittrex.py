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

    while True:

        try:
            # collect price every 5 minutes
            time.sleep(300)


            price, price_datetime = pull_data('USDT-BTC')

            print('appending bitcoin price at {}'.format(datetime.now()))
            bittrexbtcprices.append_data(parameter='btcprice', data=[price], times=[price_datetime])


        except Exception as e:
            print(e)




