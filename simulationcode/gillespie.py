# © 2024 Rebecca J. Rousseau and Justin B. Kinney
# Algebraic and diagrammatic methods for the rule-based modeling of multi-particle complexes

import numpy as np
import pandas as pd


### Define Operator, Mode, Factory, Trajectory, Constructor, Recorder classes
### (model-agnostic)
###

# Individual operators appear in products
class Operator:

    def __init__(self, family, op_type):
        self.name = family.name
        self.index = family.index
        self.op_type = op_type
        self.key = (self.name, self.index, self.op_type)
        self.family = family
        self.eligible = None
        if isinstance(self.index, tuple):
            self.index_set = set(self.index)
        else:
            self.index_set = {self.index}

    def __eq__(self, other):
        return (self.name == other.name) & \
               (self.index == other.index) & \
               (self.op_type == other.op_type)


# The operator family keeps track of eligibility
class Mode:

    def __init__(self, name, index):

        # Set name and index
        self.name = name
        self.index = index
        self.key = (self.name, self.index)

        # Set family members; initialize eligibility to ground state
        self.op_dict = {
            'hat': Operator(family=self, op_type='hat'),
            'check': Operator(family=self, op_type='check'),
            'bar': Operator(family=self, op_type='bar'),
            'tilde': Operator(family=self, op_type='tilde')
        }

        # Define operators attribute
        self.operators = tuple(self.op_dict.values())

    def set_eligibility_to_vacuum_state(self):
        for op_type, op in self.op_dict.items():
            if op_type in {'hat', 'tilde'}:
                op.eligible = True
            elif op.op_type in {'check', 'bar'}:
                op.eligible = False
            else:
                assert False, 'Should not happen.'

    def __eq__(self, other):
        return self.key == other.key


class Factory:

    def __init__(self, name, rate, operators):
        self.operators = operators
        self.rate = rate
        self.name = name


class Trajectory:

    def __init__(self, factory_list):
        #self.factory_list = []
        #self.time_list = []
        #self.time = 0
        self.available_factory_list = factory_list

        # Compute dictionary of operator families, indexed by key
        self.op_families_dict = {}
        for factory in self.available_factory_list:
            for op in factory.operators:
                self.op_families_dict[op.family.key] = op.family

    def initialize_to_vacuum_state(self):
        for op_family in self.op_families_dict.values():
            op_family.set_eligibility_to_vacuum_state()

    #
    # ALGORITHM 1
    #
    def run(self, recorder, verbose=True):

        # Set num_steps
        self.recorder = recorder
        self.num_steps = recorder.num_steps
        self.step_num = 0
        self.record_every = recorder.record_every

        # Initialize all operators to vacuum state
        self.initialize_to_vacuum_state()

        # Create fresh state constructor
        self.constructor = Constructor()

        #  Create recorder object
        self.recorder.initialize()

        # Initialize time
        t = 0

        # Iterate over time points
        for self.step_num in range(self.num_steps):

            # Get list of eligible comp ops
            eligible_factory_list = [factory for factory in
                                    self.available_factory_list
                                    if is_factory_eligible(factory)]

            # Take Gillespie step
            factory, delta_t = gillespie_step(eligible_factory_list)

            # Increment time
            t += delta_t

            # Add to constructor
            for op in factory.operators:

                # If hat operator, append to list, and flip eligibility of all
                # operator family members
                if op.op_type == 'hat':
                    self.constructor.operator_list.append(op)
                    flip_mode_excitation(op.family)

                # If check operator, remove from list, and flip eligibility of
                # all operator family members
                elif op.op_type == 'check':
                    conj_op = op.family.op_dict['hat']
                    self.constructor.operator_list.remove(conj_op)
                    flip_mode_excitation(op.family)

            # Step trajectory forward
            #self.take_step(factory=factory, delta_t=delta_t)

            # Update summary stats
            if self.step_num % self.record_every == 0:

                # TODO: Record step number
                self.recorder.record(time=t,
                                     state_cons=self.constructor)

                # Update user
                if verbose:
                    print('.', end='')


def flip_mode_excitation(mode):
    """
    Flips the excitation value a mode and all 4 mode operators
    :param mode:
    :return: None
    """
    for op in mode.operators:
        op.eligible = not op.eligible


def is_factory_eligible(factory):
    """
    Gives a factory, computes whether the factory is eligible to be applied to
    the current state.
    :param factory: (Factory object)
    :return: factory_eligibility (bool)
    """
    factory_eligibility = all([op.eligible for op in factory.operators])
    return factory_eligibility


def gillespie_step(eligible_factories):
    """
    Given a set of eligible factories, chooses one factory and a time step
    :param eligible_factories:
    :return: (factory, delta_t)
    """
    # Compute rates
    rates = np.array([factory.rate for factory in eligible_factories])
    total_rate = sum(rates)
    probs = rates / total_rate

    # Choose factory and time step
    delta_t = np.random.exponential(scale=1 / total_rate)
    factory = np.random.choice(eligible_factories, p=probs)

    return factory, delta_t


class Constructor:
    def __init__(self):
        self.operator_list = []

    def get_indices_in_complex(self, indices_in_complex):

        # Get all ops that contain an index in indices_in_complex:
        new_indices_in_complex = set(indices_in_complex).copy()

        # For each operator
        for op in self.operator_list:

            # If op indices interset new indices
            if new_indices_in_complex.intersection(op.index_set):
                # Add new indices in op.index_set to new_indices_in_complex
                new_indices_in_complex = new_indices_in_complex.union(
                    op.index_set)

        # If any new indices have been added, iterate
        if len(new_indices_in_complex) > len(indices_in_complex):
            return self.get_indices_in_complex(new_indices_in_complex)

        # If no new indices have been added, return
        else:
            return indices_in_complex

    def get_index_sets_of_all_complexes(self):

        # Get list of all indices
        indices_to_process = set({})
        for op in self.operator_list:
            indices_to_process = indices_to_process.union(op.index_set)

        index_set_list = []
        while len(indices_to_process) > 0:
            anchor_index = next(iter(indices_to_process))
            indices_in_complex = self.get_indices_in_complex({anchor_index})
            index_set_list.append(indices_in_complex)
            indices_to_process = indices_to_process.difference(
                indices_in_complex)

        return index_set_list

    def get_ops_lists_for_all_complexes(self):

        # Get index sets of all complexes
        index_sets = self.get_index_sets_of_all_complexes()

        # Get unique set of indices
        all_indices = set({})
        for index_set in index_sets:
            all_indices = all_indices.union(index_set)

        # Next, create dict mapping each individual index to a set of operators
        index_to_ops_dict = {index: [] for index in all_indices}
        for op in self.operator_list:
            for index in op.index_set:
                index_to_ops_dict[index].append(op)

        # Finally, make lists of op lists
        ops_lists = []
        for index_set in index_sets:
            ops_list = []
            for index in index_set:
                ops_list.extend(index_to_ops_dict[index])

            # Make ops_list unique:
            ops_dict = {op.key: op for op in ops_list}
            ops_list = list(ops_dict.values())

            ops_lists.append(ops_list)
        return ops_lists


# Create container class for summary stats
class Recorder:

    def __init__(self,
                 num_steps,
                 record_every,
                 fields_to_record,
                 complexes_to_record,
                 get_complex_name_fn):

        # Set number of records
        self.num_steps = num_steps
        self.record_every = record_every
        self.num_records = int(np.floor(num_steps / record_every))

        # Set names of fields and complexes
        self.fields_to_record = fields_to_record.copy()
        self.complexes_to_record = complexes_to_record.copy()

        # Function to count complexes from a list of operator lists
        self.get_complex_name_fn = get_complex_name_fn

    def initialize(self):
        # Initialize record_num and time
        self.record_num = 0
        self.time = 0.0

        # Dataframe in which to hold field counts
        self.field_counts_df = pd.DataFrame(
            columns=['time'] + self.fields_to_record,
            index=range(self.num_records),
            data=0).astype({'time': float})

        # Dataframe in which to hold complex counts
        self.complex_counts_df = pd.DataFrame(
            columns=['time'] + self.complexes_to_record,
            index=range(self.num_records),
            data=0).astype({'time': float})

    # Record summary stats 
    def record(self, time, state_cons):

        # Set time
        self.time = time

        # Define Series in which to record stats for self.field_counts_df
        field_count_updates = pd.Series(index=self.fields_to_record, data=0.0)
        field_count_updates.name = self.record_num
        field_count_updates['time'] = self.time

        # Record numbers of each type of field
        for field_name in self.fields_to_record:
            field_count_updates[field_name] = sum(
                [op.name == field_name for op in state_cons.operator_list])

        # Save field_counts records
        self.field_counts_df = self.field_counts_df.transpose(copy=False)
        self.field_counts_df.update(field_count_updates)
        self.field_counts_df = self.field_counts_df.transpose(copy=False)

        # Define Series in which to record stats for self.complex_counts_df
        complex_count_updates = pd.Series(index=self.complexes_to_record,
                                          data=0.0)
        complex_count_updates.name = self.record_num
        complex_count_updates['time'] = self.time

        # Count complexes
        op_lists = state_cons.get_ops_lists_for_all_complexes()
        complex_names = []
        for op_list in op_lists:
            complex_name = self.get_complex_name_fn(op_list)
            complex_names.append(complex_name)
        complex_counts_dict = pd.value_counts(complex_names).to_dict()
        for complex_name in self.complexes_to_record:
            complex_count_updates[complex_name] = complex_counts_dict.get(
                complex_name, 0)

        # Save records
        self.complex_counts_df = self.complex_counts_df.transpose(copy=False)
        self.complex_counts_df.update(complex_count_updates)
        self.complex_counts_df = self.complex_counts_df.transpose(copy=False)

        # Increment record number
        self.record_num += 1


### Define get_operator_list and get_factory_list classes (Simplifies user interface)
### (model-agnostic)
###

def get_operator_list(field_name, indices):
    op_family_list = [Mode(field_name, index) for index in indices]
    op_list = [op for op_family in op_family_list for op in
               op_family.op_dict.values()]
    return op_list


def get_factory_list(op_list, rate, spec, indices):
    # Creat operator dict
    op_dict = {op.key: op for op in op_list}

    # Cast all indices as tuples
    if isinstance(indices[0], int):
        index_tuples = [(index) for index in indices]
    elif isinstance(indices[0], tuple):
        index_tuples = indices
    else:
        assert False, "This shouldn't happen"

    # Parse spec string
    op_specs = spec.split()

    # Get protokeys for operators
    # a protokey is a tuple: (field_name, index_name, op_type)
    # Also get names of indices
    protokeys = []
    unique_ordered_index_names = []
    for op_spec in op_specs:
        field_name, index_name, op_type = op_spec.split('_')
        protokey = (field_name, index_name, op_type)
        protokeys.append(protokey)
        index_names = list(index_name)
        for index_name in index_names:
            if index_name not in unique_ordered_index_names:
                unique_ordered_index_names.append(index_name)

    # Create dict mapping index_name to ix
    num_indices = len(unique_ordered_index_names)

    # List of transitions to ultimately return to user
    factory_list = []

    # For each index
    for factory_index in indices:

        # Convert index to tuple
        if isinstance(factory_index, int):
            factory_index = tuple([factory_index])
        elif isinstance(factory_index, tuple):
            pass;
        else:
            assert False, "This shouldn't happen"

        # Make dictionary mapping each index letter to a specific index values
        index_letter_to_ix_dict = {index_name: factory_index[i]
                                   for i, index_name in
                                   enumerate(unique_ordered_index_names)}
        op_keys = []
        for field_name, index_name, op_type in protokeys:

            # Make operator index based on index name and value of index
            op_index = tuple([index_letter_to_ix_dict[i_name] for i_name in
                              list(index_name)])
            if len(op_index) == 1:
                op_index = op_index[0]

            # Make operator key and append
            op_key = (field_name, op_index, op_type)
            op_keys.append(op_key)

        # Create list of operators based on op_keys
        ops = [op_dict[op_key] for op_key in op_keys]

        # Create factory and append to list
        factory = Factory(name=field_name, rate=rate, operators=ops)
        factory_list.append(factory)

    # Return list of factory
    return factory_list
