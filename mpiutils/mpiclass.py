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
        split_length = int(np.floor(length / self.size))
        splits = np.zeros(self.size+1, dtype='int')
        splits[1:] = split_length
        remainder = length - split_length*self.size
        splits[1:remainder] += 1
        splits = np.cumsum(splits)
        split1 = splits[:-1]
        split2 = splits[1:]
        return split1, split2


    def mpi_print(self, string):
        """Prints out a string using flush so it prints out immediately in an MPI
        setting."""
        print(string, flush=True)


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


    def end(self):
        """Ends MPI environment."""
        mpi.Finalize()
