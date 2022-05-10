import math
import mmh3
from bitarray import bitarray
from typing import Hashable


class BloomFilter(object):
    """Bloom filter using murmur3 hash function.

    Parameters
    ----------
    items_count : int
        Number of items expected to be stored in bloom filter.

    size : int
        Size of bit array to use.

    number_of_ones : int
        Number of hash functions to use.
    """
    def __init__(self, items_count: int, size: int, number_of_ones: int = -1):
        # False posible probability in decimal
        self.fp_prob = 0.05

        # Size of bit array to use
        # self.size = self.get_size(items_count, fp_prob)
        self.size = size

        # number of hash functions to use
        if number_of_ones == -1:
            self.hash_count = self.get_hash_count(self.size, items_count)
        else:
            self.hash_count = number_of_ones

            # Bit array of given size
        self.bit_array = bitarray(self.size)

        # initialize all bits as 0
        self.bit_array.setall(0)

    def add(self, item: Hashable) -> None:
        """Add one item to the filter.

        Parameters
        ----------
        item : Hashable
            Item to add to the filter.
        """
        digests = []
        for i in range(self.hash_count):
            # create digest for given item.
            # i work as seed to mmh3.hash() function
            # With different seed, digest created is different
            digest = mmh3.hash(item, i) % self.size
            digests.append(digest)
            # set the bit True in bit_array
            self.bit_array[digest] = True


    def check(self, item: Hashable) -> bool:
        """Check for existence of one item in the filter.

        Parameters
        ----------
        item : Hashable
            Item to check existence for.

        Returns
        -------
        bool
            True if item exists in filter. False otherwise.
        """
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if self.bit_array[digest] == False:
                # if any of bit is False then,its not present
                # in filter
                # else there is probability that it exist
                return False
        return True

    @classmethod
    def get_size(cls, n: int, p: float) -> int:
        """Returns the size of bit array(m) to be used.

        Following formula is used:
        m = -(n * lg(p)) / (lg(2)^2)

        Parameters
        ----------
        n : int
            Number of items expected to be stored in filter.

        p : float
            False Positive probability in decimal.

        Returns
        -------
        int
            Size of bit array(m).
        """
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @classmethod
    def get_hash_count(cls, m: int, n: int) -> int:
        """Return the hash function(k) to be used.

        Following formula is used:
        k = (m/n) * lg(2)

        Parameters
        ----------
        m : int
            Size of bit array.
        n : int
            Number of items expected to be stored in filter.

        Returns
        -------
        int
            Hash function(k).
        """
        k = (m / n) * math.log(2)
        return int(k)


