from datastorage import DataCollector
from scipy import interpolate
import matplotlib.pyplot as plt
from process_data import convert_time_to_unix
import numpy as np


def retrieve_data(filename, parameter, indexmax):

    pricedata = DataCollector(filename=filename, overwrite=False)
    prices = pricedata.retrieve_data(parameter=parameter, indexes=(0, indexmax))
    price_times = pricedata.retrieve_data(parameter=parameter+'_time', indexes=(0, indexmax))
    return prices, price_times


def linearize_data(prices, times, globaltime):

    interp = interpolate.interp1d(times, prices)
    prices_linear = interp(globaltime)
    return prices_linear



def get_linear_data(dt):

    # index all of the data
    indexmax = 999999

    btcprices, btcprice_times = retrieve_data('bittrexbtcprices', 'btcprice', indexmax)
    tetherprices, tetherprice_times = retrieve_data('bittrextetherprices', 'tetherprice', indexmax)
    ethereumprices, ethereumprice_times = retrieve_data('ethereumprices', 'ethereumprice', indexmax)
    bchprices, bchprice_times = retrieve_data('bchprices', 'bchprice', indexmax)


    # convert the times to unix time
    btcprice_times = [convert_time_to_unix(datetime.replace('T', ' ').split('.')[0]) for datetime in btcprice_times]
    tetherprice_times = [convert_time_to_unix(datetime.replace('T', ' ').split('.')[0]) for datetime in tetherprice_times]
    ethereumprice_times = [convert_time_to_unix(datetime.replace('T', ' ').split('.')[0]) for datetime in ethereumprice_times]
    bchprice_times = [convert_time_to_unix(datetime.replace('T', ' ').split('.')[0]) for datetime in bchprice_times]

    # convert the prices to floats
    btcprices = [float(price) for price in btcprices]
    tetherprices = [float(price) for price in tetherprices]
    ethereumprices = [float(price) for price in ethereumprices]
    bchprices = [float(price) for price in bchprices]


    # find the most recent time, and make all the other datasets have their most recent price at that time
    maxtime = np.max(np.array([btcprice_times[-1], tetherprice_times[-1], ethereumprice_times[-1], bchprice_times[-1]]))
    mintime = np.min(np.array([btcprice_times[0], tetherprice_times[0], ethereumprice_times[0], bchprice_times[0]]))

    for price, time in zip([btcprices, tetherprices, ethereumprices, bchprices],
                           [btcprice_times, tetherprice_times, ethereumprice_times, bchprice_times]):

        # extend the times to match the newest time
        if time[-1] < maxtime:
            time.append(maxtime)
            price.append(price[-1])

        # extend the times to match the oldest time
        if time[0] > mintime:
            time.insert(0, mintime)
            price.insert(0, price[0])


    # define a global time axis
    globaltime = np.arange(mintime, maxtime, dt)

    # linearize the data
    btcprices = linearize_data(prices=btcprices, times=btcprice_times, globaltime=globaltime)
    tetherprices = linearize_data(prices=tetherprices, times=tetherprice_times, globaltime=globaltime)
    ethereumprices = linearize_data(prices=ethereumprices, times=ethereumprice_times, globaltime=globaltime)
    bchprices = linearize_data(prices=bchprices, times=bchprice_times, globaltime=globaltime)

    pricedata = {}
    pricedata['btcprices'] = btcprices
    pricedata['tetherprices'] = tetherprices
    pricedata['ethereumprices'] = ethereumprices
    pricedata['bchprices'] = bchprices

    return globaltime, pricedata


if __name__ == "__main__":
    globaltime, pricedata = get_linear_data(dt=60*60)
    plt.plot(globaltime, pricedata['btcprices'])
    plt.plot(globaltime, pricedata['tetherprices'])
    plt.plot(globaltime, pricedata['ethereumprices'])
    plt.plot(globaltime, pricedata['bchprices'], color='blue', linewidth=3)

    plt.show()


