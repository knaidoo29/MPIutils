import numpy as np
from mpi4py import MPI as mpi

from . import loops


class MPI:


    def __init__(self):
        """Initialises MPI."""
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.loop_size = None
        self.mpi_info = 'Proc ' + str(self.rank+1)+' of ' + str(self.size)


    def wait(self):
        """Makes all jobs wait so they are synchronised."""
        self.comm.Barrier()


    def set_loop(self, loop_size):
        """Sets the size of a distributed loop."""
        self.loop_size = loop_size
        return loops.get_MPI_loop_size(loop_size, self.size)


    def mpi_ind2ind(self, mpi_ind):
        """Converts the MPI_ind of a distributed loop to the index of a full loop."""
        return loops.MPI_ind2ind(mpi_ind, self.rank, self.size, self.loop_size)


    def clean_loop(self):
        """Gets ride of loop_size definition."""
        self.loop_size = None


    def split(self, length):
        """For splitting an array across nodes."""
        split_equal = length/self.size
        split_floor = np.floor(split_equal)
        split_remain = split_equal - split_floor
        counts = split_floor*np.ones(self.size)
        counts[:int(np.round(split_remain*self.size, decimals=0))] += 1
        counts = counts.astype('int')
        splits = np.zeros(self.size+1, dtype='int')
        splits[1:] = np.cumsum(counts)
        split1 = splits[:-1]
        split2 = splits[1:]
        return split1, split2


    def split_array(self, array):
        """Returns the values of the split array."""
        split1, split2 = self.split(len(array))
        return array[split1[self.rank]:split2[self.rank]]


    def check_partition(self, NDshape, NDshape_split):
        """Returns bool array showing which axes the array is being split."""
        return np.array([NDshape[i] == NDshape_split[i] for i in range(len(NDshape))])


    def create_split_ndarray(self, arrays_nd, whichaxis):
        """Split a list of arrays based on the data partitioning scheme."""
        split_arrays = []
        for i in range(0, len(arrays_nd)):
            _array = arrays_nd[i]
            if not whichaxis[i]:
                _array = self.split_array(_array)
                split_arrays.append(_array)
            else:
                split_arrays.append(_array)
        return split_arrays


    def create_split_ndgrid(self, arrays_nd, whichaxis):
        """Create a partitioned gridded data set."""
        split_arrays = self.create_split_ndarray(arrays_nd, whichaxis)
        split_grid = np.meshgrid(*split_arrays, indexing='ij')
        return split_grid


    def mpi_print(self, *value):
        """Prints out using flush so it prints out immediately in an MPI
        setting."""
        print(*value, flush=True)


    def send(self, data, to_rank=None, tag=11):
        """Sends data from current core to other specified or all cores.

        Parameters
        ----------
        data : array
            Data to send.
        to_rank : int, optional
            Specify rank to send data to, or leave as None to send to all cores.
        tag : int, optional
            Sending tag to ensure the right data is being transfered.
        """
        if to_rank is not None:
            self.comm.send(data, dest=to_rank, tag=tag)
        else:
            for i in range(0, self.size):
                if i != self.rank:
                    self.comm.send(data, dest=i, tag=tag)


    def recv(self, source, tag=11):
        """Receive data from another node.

        Parameters
        ----------
        source : int
            Source of the data.
        tag : int
            Sending tag to ensure the right data is being transfered.

        Returns
        -------
        data : array
            Data received.
        """
        data = self.comm.recv(source=source, tag=tag)
        return data


    def collect(self, data):
        """Collects a distributed data to the processor with rank=0.

        Parameters
        ----------
        data : array
            distributed data set.
        """
        if self.rank == 0:
            datas = [data]
            for i in range(1, self.size):
                _data = self.recv(i, tag=10+i)
                datas.append(_data)
            data = np.concatenate(datas)
        else:
            self.send(data, to_rank=0, tag=10+self.rank)
            data = None
        self.wait()
        return data


    def sum(self, data):
        """Sums a distributed data set to the processor with rank=0.

        Parameters
        ----------
        data : array
            distributed data set.
        """
        if self.rank == 0:
            for i in range(1, self.size):
                _data = self.recv(i, tag=10+i)
                data += _data
        else:
            self.send(data, to_rank=0, tag=10+self.rank)
            data = None
        self.wait()
        return data


    def end(self):
        """Ends MPI environment."""
        mpi.Finalize()
