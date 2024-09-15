import numpy as np
import random 

class IndexSet:
    """
    A class representing a set of indices, which can be either a regular set or its complement.

    This class is designed to efficiently handle sets of indices, particularly for use in
    stochastic simulation algorithms like the Gillespie algorithm. It supports operations
    on multi-dimensional indices and can represent either a finite set of indices or the
    complement of such a set.

    Attributes:
        elements (set): The set of indices.
        is_complement (bool): If True, represents the complement of the set.
        max_index (int): The maximum possible index value.
        index_dim (int): The dimensionality of the indices.
        increasing_indices (bool): If True, ensures indices are in increasing order.

    Methods:
        __init__: Initialize the IndexSet.
        __contains__: Check if an item is in the set.
        get_set_size: Get the size of the set.
        __repr__: String representation of the IndexSet.
        __and__: Intersection operation.
        __or__: Union operation.
        __sub__: Difference operation.
        __eq__: Equality comparison.
    """

    def __init__(self, elements=None, is_complement=False, max_index=int(1E4), index_dim=None, increasing_indices=False):
        """
        Initialize the IndexSet.

        Args:
            elements (iterable, optional): The initial set of indices. Defaults to None.
            is_complement (bool, optional): If True, represents the complement of the set. Defaults to False.
            max_index (int, optional): The maximum possible index value. Defaults to 10000.
            index_dim (int, optional): The dimensionality of the indices. If not provided, it's inferred from elements.
            increasing_indices (bool, optional): If True, ensures indices are in increasing order. Defaults to False.

        Raises:
            ValueError: If neither elements nor index_dim is provided.

        Note:
            If increasing_indices is True, all elements must be in increasing order.
        """
        
        # Save arguments as attributes
        self.elements = set(elements or [])
        self.is_complement = is_complement
        self.max_index = max_index
        self.increasing_indices = increasing_indices

        # If elements and index_dime are provided, check that elements are consistent with index_dim
        if len(elements) > 0 and index_dim is not None:
            self.index_dim = index_dim
            assert len(list(elements)[0]) == index_dim, "Index dimension of elements must match index_dim"

        # If only elements are provided, infer index_dim from the elements
        elif len(elements) > 0 and index_dim is None:
            self.index_dim = len(list(elements)[0])

        # If only index_dim is provided, save this value
        elif len(elements) == 0 and index_dim is not None:
            self.index_dim = index_dim

        # If neither elements nor index_dim are provided, raise an error
        else:
            raise ValueError(f"Either elements={elements} or index_dim={index_dim} must be provided")
        
        # If increasing_indices is True, and elements are provided, check that elements are increasing
        if self.increasing_indices and self.elements is not None:
            for element in self.elements:
                assert element == tuple(sorted(element)), f"Elements must be increasing; element {element} is not increasing"


    def __contains__(self, item):
        """
        Check if an item is in the set.
        
        For a regular set, return True if the item is in self.elements.
        For a complement set, return True if the item is not in self.elements.
        """
        
        # If increasing indices, check that the item is in increasing order
        if self.increasing_indices:
            assert item == tuple(sorted(item)), "Elements must be increasing" 
        
        # If the set is a complement, check that the item is not in self.elements
        if self.is_complement:
            return item not in self.elements
        
        # Otherwise, check that the item is in self.elements
        else:
            return item in self.elements 
        

    def get_set_size(self):
        """
        Get the size of the set.
        
        For a regular set, return the number of elements.
        For a complement set, return infinity.
        Note that this will not always return an integer
        """
        return np.inf if self.is_complement else len(self.elements)


    def __repr__(self):
        """
        Return a string representation of the IndexSet.

        This method provides a concise and informative string representation of the IndexSet object,
        including the class name, the elements in the set, and whether it's a complement set.

        Returns:
            str: A string representation of the IndexSet object.

        Example:
            >>> index_set = IndexSet({(1, 2), (3, 4)})
            >>> repr(index_set)
            'IndexSet({(1, 2), (3, 4)}, is_complement=False)'

            >>> complement_set = IndexSet({(1, 2)}, is_complement=True)
            >>> repr(complement_set)
            'IndexSet({(1, 2)}, is_complement=True)'
        """
        return f"{self.__class__.__name__}({self.elements}, is_complement={self.is_complement})"


    def __and__(self, other):
        """Implement the & operator for intersection."""
        return self.intersection(other)


    def __or__(self, other):
        """Implement the | operator for union."""
        return self.union(other)


    def __sub__(self, other):
        """Implement the - operator for difference."""
        return self.difference(other)


    def __eq__(self, other):
        """Implement the == operator for equality."""
        return (self.elements == other.elements and 
                self.is_complement == other.is_complement)


    def __contains__(self, item):
        """
        Check if an item is in the set.
        
        For a regular set, return True if the item is in self.elements.
        For a complement set, return True if the item is not in self.elements.
        """
        if self.is_complement:
            return item not in self.elements
        else:
            return item in self.elements


    def set_to_empty(self):
        """
        Set the IndexSet to an empty set.

        This method clears all elements from the set and sets it to a regular (non-complement) empty set.

        Example:
            >>> index_set = IndexSet({(1, 2), (3, 4)})
            >>> index_set.set_to_empty()
            >>> print(index_set)
            IndexSet(set(), is_complement=False)
        """
        self.elements = set({})
        self.is_complement = False


    def complement(self):
        """
        Return the complement of the IndexSet.

        This method returns a new IndexSet object representing the complement of the current set.
        In essence it flips the is_complement flag.

        Returns:
            IndexSet: A new IndexSet object representing the complement of the current set.
        """
        return IndexSet(elements=self.elements, is_complement=(not self.is_complement), index_dim=self.index_dim)


    def union(self, other):
        """
        Return the union of the IndexSet with another IndexSet.

        This method returns a new IndexSet object representing the union of the current set and another set.
        It ensures that the resulting set has unique elements and maintains the order of elements.

        Args:
            other (IndexSet): The other IndexSet object to be unioned with the current set. 

        Returns:
            IndexSet: A new IndexSet object representing the union of the current set and the other set.
        """

        # Check that the index dimensions match
        assert self.index_dim == other.index_dim, "Index dimensions must match"

        # If both sets are complements, return the intersection of the elements and set is_complement to True
        if self.is_complement and other.is_complement:
            return IndexSet(elements=self.elements & other.elements, is_complement=True, index_dim=self.index_dim)
        
        # If self is a complement and other is not, return the difference of the elements and set is_complement to True
        elif self.is_complement:
            return IndexSet(elements=self.elements - other.elements, is_complement= True, index_dim=self.index_dim)
        
        # If other is a complement and self is not, return the reversed difference of the elements and set is_complement to True
        elif other.is_complement:
            return IndexSet(elements=other.elements - self.elements, is_complement= True, index_dim=self.index_dim)
        
        # Otherwise, return the union of the elements and set is_complement to False
        else:
            return IndexSet(elements=self.elements | other.elements, is_complement=False, index_dim=self.index_dim)


    def intersection(self, other):
        """
        Return the intersection of the IndexSet with another IndexSet.

        This method returns a new IndexSet object representing the intersection of the current set and another set.
        It ensures that the resulting set has unique elements and maintains the order of elements.

        Args:
            other (IndexSet): The other IndexSet object to be intersected with the current set. 
        """

        # Check that the index dimensions match 
        assert self.index_dim == other.index_dim, "Index dimensions must match"

        # If both sets are complements, return the union of the elements and set is_complement to True
        if self.is_complement and other.is_complement:
            return self.complement().union(other.complement()).complement()
        
        # If self is a complement and other is not, return other minus the complement of self (will have is_complement=False)
        elif self.is_complement:
            return other.difference(self.complement())
        
        # If other is a complement and self is not, return self minus the complement of other (will have is_complement=False)
        elif other.is_complement:
            return self.difference(other.complement())
        
        # Otherwise, return the intersection of the elements and set is_complement to False
        else:
            return IndexSet(elements=self.elements & other.elements, is_complement=False, index_dim=self.index_dim)


    def difference(self, other):
        """
        Return the difference of the IndexSet with another IndexSet.

        This method returns a new IndexSet object representing the difference of the current set and another set.
        It ensures that the resulting set has unique elements and maintains the order of elements.

        Args:
            other (IndexSet): The other IndexSet object to be subtracted from the current set. 
        """ 

        # Check that the index dimensions match 
        assert self.index_dim == other.index_dim, "Index dimensions must match"

        # If both sets are complements, return the difference of the elements and set is_complement to False
        if self.is_complement and other.is_complement:
            return IndexSet(elements=other.elements - self.elements, is_complement=False, index_dim=self.index_dim)
        
        # If self is a complement and other is not, then take the union of the elements and set is_complement to True
        elif self.is_complement:
            return IndexSet(elements=self.elements | other.elements, is_complement=True, index_dim=self.index_dim)
            #return self.complement().union(other).complement()
        
        # If other is a complement and self is not, then take the intersection of the elements and set is_complement to False
        elif other.is_complement:
            return IndexSet(elements=other.elements & self.elements, is_complement=False, index_dim=self.index_dim)
            #return other.complement().union(self).complement()
        
        # Otherwise, neither is a complement. Just return the difference of the elements and set is_complement to False
        else:
            return IndexSet(elements=self.elements - other.elements, is_complement=False, index_dim=self.index_dim)
        

    def sample_index(self, max_num_tries=100):
        """
        Sample a random element from the IndexSet.

        This method samples a random element from the IndexSet, ensuring that the sampled element is not already in the set.
        It uses a maximum number of tries to avoid infinite loops in case of failure.

        Args:
            max_num_tries (int, optional): The maximum number of tries to sample an element. Defaults to 100.
        """

        # If the set is empty, return None
        if self.get_set_size() == 0:
            return None

        # Otherwise, if the set is not a complement, return a randomly selected element from the list of elements
        elif not self.is_complement:

            # Return a randomly selected element from the list of elements
            return random.choice(list(self.elements))
        
        # Otherwise, the set is a complement and the sampled element must be randomly generated
        else:
            
            # Try up to max_num_tries to sample a random element
            for _ in range(max_num_tries):

                # Sample random indices; package into candidate tuple
                candidate = tuple(random.randrange(self.max_index) for i in range(self.index_dim))

                # If restrict to increasing indices
                if self.increasing_indices:

                    # If elements in candidate are not unique, start loop again
                    if len(set(candidate)) != self.index_dim:
                        continue
                    
                    # Otherwise, sort the candidate 
                    else:
                        candidate = tuple(sorted(candidate))

                # If candidate is not in the set, return it
                if candidate not in self.elements:
                    return candidate

            # If we get here, we failed to sample an element from the complement of the set
            raise ValueError(f"Could not sample from complement of index set {self} within {max_num_tries} tries")
        


    def sample_indices(self, num_indices, max_num_tries=100):
        """
        Sample a specified number of random elements from the IndexSet.

        This method samples a specified number of random elements from the IndexSet, ensuring that the sampled elements are not already in the set.
        It uses a maximum number of tries to avoid infinite loops in case of failure.

        Args:
            num_indices (int): The number of indices to sample.
            max_num_tries (int, optional): The maximum number of tries to sample an element. Defaults to 100.
        """

        # If the set is empty, return None
        if self.get_set_size() < num_indices:
            return None

        # Otherwise, if the set is not a complement, return a randomly selected element from the list of elements
        elif not self.is_complement:

            # Return a randomly selected element from the list of elements
            return random.sample(list(self.elements), num_indices)
        
        # Otherwise, the set is a complement and the sampled element must be randomly generated
        else:
            
            # Initialize list to hold candidate indices
            candidates = []

            # Try up to max_num_tries to sample a random element
            for _ in range(max_num_tries):

                # Sample random indices; package into candidate tuple
                candidate = tuple(random.randrange(self.max_index) for i in range(self.index_dim))

                # If restrict to increasing indices
                if self.increasing_indices:

                    # If elements in candidate are not unique, start loop again
                    if len(set(candidate)) != self.index_dim:
                        continue
                    
                    # Otherwise, sort the candidate 
                    else:
                        candidate = tuple(sorted(candidate))

                # If candidate is not in the set, return it
                if candidate not in self.elements and candidate not in candidates:
                    candidates.append(candidate)

            # If we have enough candidates, return them
            if len(candidates) == num_indices:
                return candidates
            
            # If we get here, we failed to sample an element from the complement of the set
            else: 
                raise ValueError(f"Could not sample {num_indices} indices from complement of index set {self} within {max_num_tries} tries")
            
