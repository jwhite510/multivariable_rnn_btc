from datastorage import DataCollector



indexmax = 999999



# retrieve the data
bittrexbtcpricedata = DataCollector(filename='bittrexbtcprices', overwrite=False)
btcprices = bittrexbtcpricedata.retrieve_data(parameter='btcprice', indexes=(0, indexmax))
btcprice_times = bittrexbtcpricedata.retrieve_data(parameter='btcprice_time', indexes=(0, indexmax))


print(len(btcprices))
index = -1

# view the submissions
print(btcprices[index])
print(btcprice_times[index])






























