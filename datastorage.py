import tables
import os



class RedditData():

    def __init__(self, filename, parameters=None, overwrite=False, timecharlength=30, checklength=False):


        self.filename = filename + '.hdf5'
        self.parameters = parameters
        self.timecharlength = timecharlength
        self.checklength = checklength

        # check if the file already exists
        exists = os.path.exists('./{}'.format(self.filename))


        # if not, create it
        if not exists or overwrite:
            print('creating hdf5 file {}'.format(self.filename))
            # create file
            hdf5_file = tables.open_file(self.filename, mode='w')

            for parameter in self.parameters:

                max_length = parameter['maxlength']
                name = parameter['name']

                hdf5_file.create_earray(hdf5_file.root, name, tables.StringAtom(itemsize=max_length), shape=(1, 0))
                hdf5_file.create_earray(hdf5_file.root, name+'_time',
                                        tables.StringAtom(itemsize=self.timecharlength), shape=(1, 0))

            hdf5_file.close()

        else:
            pass


    def append_data(self, parameter, data, times):

        # append the file
        hdf5_file = tables.open_file(self.filename, mode='a')

        if self.checklength:
            # check if the data is too long
            max_length=None
            for param in self.parameters:
                if param['name'] == parameter:
                    max_length = param['maxlength']

            for data_entry in data:
                if len(data_entry) > max_length:
                    print('data exceeded allowable length, max:{}, data:{}'.format(max_length,
                                                                                        len(data_entry)))


        # append the data
        if len(data) == len(times):
            getattr(hdf5_file.root, parameter).append([data])
            getattr(hdf5_file.root, parameter+'_time').append([times])

        hdf5_file.close()


    def retrieve_data(self, parameter, indexes):


        hdf5_file = tables.open_file(self.filename, mode='r')

        values = getattr(hdf5_file.root, parameter)[0, indexes[0]:indexes[1]]
        hdf5_file.close()

        values_str = [element.decode('UTF-8') for element in values]

        return(values_str)
