import numpy as np
import random
import time
from tqdm.notebook import tqdm
from collections.abc import Iterable
from index_set import IndexSet


class Simulation:
    def __init__(self, space, rules, custom_stats_func=None):
        self.space = space
        self.rules = rules
        self.t = None
        self.step_info = None
        self.num_steps = None
        self.custom_stats_func = custom_stats_func
        self.custom_stat_names = None
        self.rule_to_apply = None

    def initialize_state(self):
        self.space.initialize_to_vacuum()
        self.rules.update_eligibility()
        self.t = 0
        self.step_info = []
        self.num_steps = 0

    def run(self, num_steps, record_every=1):
        # Perform Gillespie steps
        num_rules = len(self.rules.rules)
        for step_num in tqdm(range(num_steps), desc="Progress"):

            # For recording performance
            rest_start_time = time.perf_counter()

            # Get list of rule-specific rates
            rule_rates = np.array([R.eligible_rate for R in self.rules.rules])

            # Compute the total rate
            tot_rate = sum(rule_rates)

            # Draw a time step and increment t
            delta_t = np.random.exponential(1 / tot_rate)
            self.t += delta_t

            # Draw a random rule
            rule_probs = rule_rates / tot_rate
            rule_num = \
            random.choices(range(num_rules), weights=rule_probs, k=1)[0]
            self.rule_to_apply = self.rules.rules[rule_num]

            # Apply the rule
            self.rule_to_apply.apply()

            # For recording performance
            rest_duration = time.perf_counter() - rest_start_time
            update_start_time = time.perf_counter()

            # Update rule eligibility; this is the slow step
            self.rules.update_eligibility()

            # For recording performance
            update_duration = time.perf_counter() - update_start_time

            # Record all step info and append to list
            if (self.num_steps % record_every) == 0:
                # Store step info
                step_info_dict = dict(step_num=self.num_steps,
                    R=self.rule_to_apply,
                    t=self.t,
                    delta_t=delta_t,
                    rule_num=rule_num,
                    tot_rate=tot_rate,
                    rule_probs=rule_probs,
                    rest_duration=rest_duration,
                    update_duration=update_duration,
                    num_eligible_indices=np.array([R.num_eligible_indices for
                                                   R in self.rules.rules]),
                    eligible_rates = np.array([R.eligible_rate for
                                               R in self.rules.rules])
                )
                # If a custom stats func is defined, add dict elements to step_info
                if self.custom_stats_func:
                    custom_stats = self.custom_stats_func()
                    step_info_dict |= custom_stats
                    if step_num==0:
                        self.custom_stat_names = list(custom_stats.keys())

                 # Store step_info
                self.step_info.append(step_info_dict)

            self.num_steps += 1


class FockSpace:
    def __init__(self, fields=()):
        self.fields = []
        self.field_dict = {}
        for field in fields:
            self.add_field(field)

    def add_field(self, field):
        self.fields.append(field)
        self.field_dict[field.name] = field

    def initialize_to_vacuum(self):
        for field in self.fields:
            field.indices = set({})

    def report_state(self):
        for field in self.fields:
            indices = field.indices
            if field.index_dim == 1:
                indices = set(i[0] for i in indices)
            print(f'{field.name}: {indices}')
        print('------------')


class RuleSet:
    def __init__(self, rules):
        self.rules = rules

    def update_eligibility(self):
        for R in self.rules:
            R.compute_eligible_indices()


class Field:
    def __init__(self, name, index_dim, index_constraint=None):
        self.name = name
        self.indices = set()
        self.index_constraint = index_constraint
        self.index_dim = index_dim

    def validate_index(self, index):
        if isinstance(index, int):
            index = (index,)
        elif isinstance(index, Iterable):
            index = tuple(index)
        else:
            assert False, f'index={index} is of invalid type={type(index)}'

        assert len(index) == self.index_dim, \
            f'dimension of index {index} is {len(index)}; must be {self.index_dim}'
        if self.index_constraint:
            assert self.index_constraint(index), \
                f'index {index} does not satisfy constraint.'
        return index

    def excite(self, index):
        index = self.validate_index(index)

        # Check for errors
        assert index not in self.indices, \
            f'index {index} is already excited.'
        assert len(index) == self.index_dim, \
            f'dimension of index {index} is {len(index)}; must be {self.index_dim}'
        if self.index_constraint:
            assert self.index_constraint(index), \
                f'index {index} does not satisfy constraint.'

        # Excite index
        self.indices.add(index)

    def relax(self, index):
        index = self.validate_index(index)

        # Check for errors
        assert index in self.indices, \
            f'index {index} is not excited.'

        # Relax index
        self.indices.remove(index)


class FieldOp:
    def __init__(self, field, op_type):
        assert isinstance(field, Field)
        self.field = field
        assert op_type in ('hat', 'bar', 'check', 'tilde')
        self.op_type = op_type
        self.name = f'{self.field.name}_{self.op_type}'
        self.index_dim = self.field.index_dim

    def apply(self, index):
        index = self.field.validate_index(index)
        match self.op_type:
            case 'hat':
                if not index in self.field.indices:
                    self.field.excite(index)
                else:
                    assert False, f'Applying {self.name} with index {index} kills state.'
            case 'check':
                if index in self.field.indices:
                    self.field.relax(index)
                else:
                    assert False, f'Applying {self.name} with index {index} kills state.'
            case 'bar':
                assert index in self.field.indices, \
                    f'Applying {self.name} with index {index} kills state.'
            case 'tilde':
                assert index not in self.field.indices, \
                    f'Applying {self.name} with index {index} kills state.'

    def get_conjugate_op(self):

        # Determine conjugate op type
        match self.op_type:
            case 'hat':
                conj_op_type='check'
            case 'check':
                conj_op_type='hat'
            case _:
                conj_op_type=self.op_type
                
        # Return conjugate op
        return FieldOp(field=self.field, op_type=conj_op_type)


class Rule:
    def __init__(self, name, rate, field_ops, index_dim, index_spec,
                 index_constraint=None):
        # String name of rule
        self.name = name

        # Rate at which rule is applied
        self.rate = rate

        # Number of indices passed to rule
        self.index_dim = index_dim

        # Corresponding fields
        self.field_ops = field_ops

        # A tuple of tuple of numbers indicating which among the index_dim indices
        # passed to the rule will be given to which fields
        assert len(index_spec) == len(field_ops), \
            f'len(index_spec)={len(index_spec)} does not match len(field_ops)={len(field_ops)}'
        # Convert each entry in index_spec to tuple if need be
        index_spec = list(index_spec)
        for i in range(len(index_spec)):
            poss = index_spec[i]
            if not isinstance(poss, Iterable):
                index_spec[i] = (poss,)
            else:
                index_spec[i] = tuple(poss)
        index_spec = tuple(index_spec)
        for field_index_spec in index_spec:
            for pos in field_index_spec:
                assert isinstance(pos, int)
                assert pos >= 0
                assert pos < self.index_dim, f'pos={pos} is not less than self.index_dim={self.index_dim}'
        self.index_spec = index_spec
        self.index_constraint = index_constraint

        self.eligible_index = None
        self.eligible_rate = None
        self.num_eligible_indices = None

    def validate_index(self, index):
        if isinstance(index, int):
            index = (index,)
        elif isinstance(index, Iterable):
            index = tuple(index)
        else:
            assert False, f'index={index} is of invalid type={type(index)}'

        assert len(index) == self.index_dim, \
            f'dimension of index {index} is {len(index)}; must be {self.index_dim}'
        if self.index_constraint:
            assert self.index_constraint(index), \
                f'index {index} does not satisfy constraint.'
        return index

    def apply_to_index(self, index):
        rule_index = self.validate_index(index)
        for k in range(len(self.field_ops)):
            poss = self.index_spec[k]
            op = self.field_ops[k]
            op_index = tuple(rule_index[pos] for pos in poss)
            op.apply(op_index)

    def apply(self):
        assert not (
                    self.eligible_index is None), f'rule {self.name} is not eligible'
        self.apply_to_index(self.eligible_index)


class MonomerCreationRule(Rule):
    def __init__(self, name, rate, particle, sites=None):
        self.particle = particle
        self.sites = sites
        field_ops = [FieldOp(field=particle, op_type='hat')]
        if self.sites is not None:
            field_ops += [FieldOp(field=site, op_type='tilde') for
                          site in self.sites]

        super().__init__(name=name,
                         rate=rate,
                         field_ops=field_ops,
                         index_dim=1,
                         index_spec=[0] * len(field_ops))

    def compute_eligible_indices(self):

        # Get set of indices for the particle field as well as all site fields
        field_indices_sets = [set(i[0] for i in self.particle.indices)]
        if self.sites is not None:
            field_indices_sets += [set(i[0] for i in site.indices) for site in
                                   self.sites]

        # The union of these indices is the set of inelligible indices
        inelligible_indices = set.union(*field_indices_sets)
        self.num_eligible_indices = np.inf

        # Set eligible_index to be the smallest eligible index.
        # This is fine; doesn't need to be random here.
        if len(inelligible_indices) > 0:
            self.eligible_index = max(inelligible_indices) + 1
        else:
            self.eligible_index = 0

        # Eligible rate is just rate
        self.eligible_rate = self.rate


class MonomerAnnihilationRule(Rule):
    def __init__(self, name, rate, particle, sites=None):
        self.particle = particle
        self.sites = sites
        field_ops = [FieldOp(field=particle, op_type='check')]
        if self.sites is not None:
            field_ops += [FieldOp(field=site, op_type='tilde') for
                          site in self.sites]

        super().__init__(name=name,
                         rate=rate,
                         field_ops=field_ops,
                         index_dim=1,
                         index_spec=[0] * len(field_ops))

    def compute_eligible_indices(self):
        # Get set of indices for the particle
        eligible_indices = set(i[0] for i in self.particle.indices)

        # Subtract indices corresponding to excited sites
        if self.sites is not None:
            site_indices_sets = [set(i[0] for i in site.indices) for site in
                                 self.sites]
            eligible_indices -= set.union(*site_indices_sets)

        # Eligible indices are the particle indices that do not have any excited site indices
        self.num_eligible_indices = len(eligible_indices)

        # Compute eligible_rate and select eligible_index
        self.eligible_rate = self.rate * len(eligible_indices)
        if self.num_eligible_indices  > 0:
            self.eligible_index = random.choice(list(eligible_indices))
        else:
            self.eligible_index = None


class HomotypicInteractionCreationRule(Rule):
    def __init__(self, name, rate, A, a, I):
        self.A = A
        self.a = a
        self.I = I
        A_bar = FieldOp(field=A, op_type='bar')
        a_hat = FieldOp(field=a, op_type='hat')
        I_hat = FieldOp(field=I, op_type='hat')
        super().__init__(name=name,
                         rate=rate,
                         field_ops=(A_bar, A_bar, a_hat, a_hat, I_hat),
                         index_dim=2,
                         index_spec=(0,1,0,1,(0,1)),
                         index_constraint=lambda i: i[0]<i[1])

    def compute_eligible_indices(self):
        # Compute eligible monomers
        monomer_indices = list(self.A.indices - self.a.indices)
        num_monomer_indices = len(monomer_indices)

        # Compute eligible rate
        self.num_eligible_indices = num_monomer_indices*(num_monomer_indices-1)/2
        self.eligible_rate = self.rate * self.num_eligible_indices

        # Choose eligible index
        if self.num_eligible_indices > 0:
            i,j = random.sample(list(monomer_indices), k=2)
            assert i != j, f"Indices i={i} and j={j} should not be equal."
            self.eligible_index = (i[0],j[0]) if i<j else (j[0],i[0])
        else:
            self.eligible_index = None


class HomotypicInteractionAnnihilationRule(Rule):
    def __init__(self, name, rate, A, a, I):
        self.A = A
        self.a = a
        self.I = I
        A_bar = FieldOp(field=A, op_type='bar')
        a_check = FieldOp(field=a, op_type='check')
        I_check = FieldOp(field=I, op_type='check')
        super().__init__(name=name,
                         rate=rate,
                         field_ops=(A_bar, A_bar, a_check, a_check, I_check),
                         index_dim=2,
                         index_spec=(0,1,0,1,(0,1)),
                         index_constraint=lambda i: i[0]<i[1])

    def compute_eligible_indices(self):
        # Compute eligible rate
        self.num_eligible_indices = len(self.I.indices)
        self.eligible_rate = self.rate * self.num_eligible_indices

        # Choose eligible index
        if self.num_eligible_indices > 0:
            self.eligible_index = random.choice(list(self.I.indices))
        else:
            self.eligible_index = None


class HeterotypicInteractionCreationRule(Rule):
    def __init__(self, name, rate, A, B, a, b, J,
                 A_sites_vacant=(), A_sites_occupied=(),
                 B_sites_vacant=(), B_sites_occupied=()):
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.J = J
        self.A_sites_vacant = A_sites_vacant
        self.A_sites_occupied = A_sites_occupied
        self.B_sites_vacant = B_sites_vacant
        self.B_sites_occupied = B_sites_occupied
        A_bar = FieldOp(field=A, op_type='bar')
        B_bar = FieldOp(field=B, op_type='bar')
        a_hat = FieldOp(field=a, op_type='hat')
        b_hat = FieldOp(field=b, op_type='hat')
        J_hat = FieldOp(field=J, op_type='hat')

        A_sites_vacant_ops = [FieldOp(field=c, op_type='tilde') for c in A_sites_vacant]
        A_sites_occupied_ops = [FieldOp(field=c, op_type='bar') for c in A_sites_occupied]
        A_conditional_spec = [0] * (len(A_sites_vacant) + len(A_sites_occupied))

        B_sites_vacant_ops = [FieldOp(field=c, op_type='tilde') for c in B_sites_vacant]
        B_sites_occupied_ops = [FieldOp(field=c, op_type='bar') for c in B_sites_occupied]
        B_conditional_spec = [1] * (len(B_sites_vacant) + len(B_sites_occupied))

        field_ops = [A_bar, B_bar, a_hat, b_hat, J_hat] \
                    + A_sites_vacant_ops + A_sites_occupied_ops \
                    + B_sites_vacant_ops + B_sites_occupied_ops
        index_spec = [0,1,0,1,(0,1)] \
                     + A_conditional_spec \
                     + B_conditional_spec

        super().__init__(name=name,
                         rate=rate,
                         field_ops=field_ops,
                         index_dim=2,
                         index_spec=index_spec)

    def compute_eligible_indices(self):
        # Compute eligible rate
        eligible_A_indices = self.A.indices - self.a.indices
        for site in self.A_sites_vacant:
            eligible_A_indices = eligible_A_indices - site.indices
        for site in self.A_sites_occupied:
            eligible_A_indices = eligible_A_indices & site.indices

        eligible_B_indices = self.B.indices - self.b.indices
        for site in self.B_sites_vacant:
            eligible_B_indices = eligible_B_indices - site.indices
        for site in self.B_sites_occupied:
            eligible_B_indices = eligible_B_indices & site.indices

        self.num_eligible_indices = len(eligible_A_indices) * len(eligible_B_indices)
        self.eligible_rate = self.rate * self.num_eligible_indices

        # Choose eligible index
        if self.num_eligible_indices > 0:
            i = random.choice(list(eligible_A_indices))
            j = random.choice(list(eligible_B_indices))
            self.eligible_index = (i[0],j[0])
        else:
            self.eligible_index = None


class HeterotypicInteractionAnnihilationRule(Rule):
    def __init__(self, name, rate, A, B, a, b, J,
                 A_sites_vacant=(), A_sites_occupied=(),
                 B_sites_vacant=(), B_sites_occupied=()):
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.J = J
        self.A_sites_vacant = A_sites_vacant
        self.A_sites_occupied = A_sites_occupied
        self.B_sites_vacant = B_sites_vacant
        self.B_sites_occupied = B_sites_occupied
        A_bar = FieldOp(field=A, op_type='bar')
        B_bar = FieldOp(field=B, op_type='bar')
        a_check = FieldOp(field=a, op_type='check')
        b_check = FieldOp(field=b, op_type='check')
        J_check = FieldOp(field=J, op_type='check')

        A_sites_vacant_ops = [FieldOp(field=c, op_type='tilde') for c in A_sites_vacant]
        A_sites_occupied_ops = [FieldOp(field=c, op_type='bar') for c in A_sites_occupied]
        A_conditional_spec = [0] * (len(A_sites_vacant) + len(A_sites_occupied))

        B_sites_vacant_ops = [FieldOp(field=c, op_type='tilde') for c in B_sites_vacant]
        B_sites_occupied_ops = [FieldOp(field=c, op_type='bar') for c in B_sites_occupied]
        B_conditional_spec = [1] * (len(B_sites_vacant) + len(B_sites_occupied))

        field_ops = [A_bar, B_bar, a_check, b_check, J_check] \
                    + A_sites_vacant_ops + A_sites_occupied_ops \
                    + B_sites_vacant_ops + B_sites_occupied_ops
        index_spec = [0,1,0,1,(0,1)] \
                     + A_conditional_spec \
                     + B_conditional_spec

        super().__init__(name=name,
                         rate=rate,
                         field_ops=field_ops,
                         index_dim=2,
                         index_spec=index_spec)

    def compute_eligible_indices(self):
        # Compute eligible A indices
        eligible_A_indices = self.A.indices & self.a.indices
        for site in self.A_sites_vacant:
            eligible_A_indices = eligible_A_indices - site.indices
        for site in self.A_sites_occupied:
            eligible_A_indices = eligible_A_indices & site.indices

        # Comput eligible B indices
        eligible_B_indices = self.B.indices & self.b.indices
        for site in self.B_sites_vacant:
            eligible_B_indices = eligible_B_indices - site.indices
        for site in self.B_sites_occupied:
            eligible_B_indices = eligible_B_indices & site.indices

        # Compute eligible index pairs
        eligible_indices = set([])
        for i,j in self.J.indices:
            if (i,) in eligible_A_indices and (j,) in eligible_B_indices:
                eligible_indices.add((i,j))
        self.eligible_indices = eligible_indices
        self.num_eligible_indices = len(eligible_indices)

        # Compute eligible rate
        self.eligible_rate = self.rate * self.num_eligible_indices

        # Choose eligible index
        if self.num_eligible_indices > 0:
            self.eligible_index = random.choice(list(eligible_indices))
        else:
            self.eligible_index = None


class Rule2:
    def __init__(self, name, rate, spec_str, fock_space):
        """
        Parameters
        ----------
        name: (str)
        rate: (float)
        spec_string: (str)
        fock_space: (FockSpace)

        Example: 
        spec_str = 'A_bar_i A_bar_j a_hat_i b_hat_j I_hat_ij'
        """

        # String name of rule
        self.name = name

        # Rate at which rule is applied
        self.rate = rate

        # Fock space
        self.fock_space = fock_space

        # Fill out field_ops list
        index_pos = 0
        self.field_ops = []
        index_spec = []
        index_pos_dict = dict()

        # Iterate over works in spec_str
        for word in spec_str.split(' '):

            # Split word 
            field_name, op_type, index_str = word.split('_')

            # If field is present in fock space, use that
            # Otherwise create new field and register in fock spack
            if field_name in fock_space.field_dict.keys():
                field = fock_space.field_dict[field_name]
            else:
                field = Field(name=field_name, index_dim=len(index_str))
                fock_space.add_field(field)

            # Iterate over indices in index_str
            this_index_spec = []
            for index_char in index_str:

                # If index_char not in index_pos_dict, add it and register 
                # position in index_spec
                if index_char not in index_pos_dict.keys():
                    index_pos_dict[index_char] = index_pos
                    index_pos += 1

                # Append index position to this_index_spec
                this_index_spec.append(index_pos_dict[index_char])

            # Store field indices
            index_spec.append(tuple(this_index_spec))
            
            # Create and store operator and add info
            op = FieldOp(field, op_type=op_type)
            op.index_name = index_str
            op.index_spec = tuple(this_index_spec)
            self.field_ops.append(op)
 
        # Store index spec
        self.index_spec = tuple(index_spec)
        self.index_pos_dict = index_pos_dict

        # Number of indices passed to rule
        self.index_dim = index_pos

        # For tracking eligibility during Gillespie simulation
        self.eligible_index = None
        self.eligible_rate = None
        self.num_eligible_indices = None


    def get_conjugate_rule(self, name, rate):

        # Compute conjugate spec string
        op_type_to_conj_op_type = {'hat':'check', 'check':'hat', 'bar':'bar', 'tilde':'tilde'}
        conj_spec_str = ' '.join([f'{op.field.name}_{op_type_to_conj_op_type[op.op_type]}_{op.index_name}' 
                                  for op in self.field_ops])
        
        # Create conjugate rule
        conjugate_rule = self.__class__(name=name,
                                        rate=rate,
                                        spec_str=conj_spec_str,
                                        fock_space=self.fock_space)

        return conjugate_rule


    def validate_index(self, index):
        if isinstance(index, int):
            index = (index,)
        elif isinstance(index, Iterable):
            index = tuple(index)
        else:
            assert False, f'index={index} is of invalid type={type(index)}'

        assert len(index) == self.index_dim, \
            f'dimension of index {index} is {len(index)}; must be {self.index_dim}'
        if self.index_constraint:
            assert self.index_constraint(index), \
                f'index {index} does not satisfy constraint.'
        return index


    def apply_to_index(self, index):
        rule_index = self.validate_index(index)
        for k in range(len(self.field_ops)):
            poss = self.index_spec[k]
            op = self.field_ops[k]
            op_index = tuple(rule_index[pos] for pos in poss)
            op.apply(op_index)


    def apply(self):
        assert not (self.eligible_index is None), f'rule {self.name} is not eligible'
        self.apply_to_index(self.eligible_index)
            

def _compute_eligible_indices(ops, index_dim):
    """
    Compute eligible indices from a list of field operators.

    This function calculates the set of eligible indices by intersecting the
    eligible indices of each field operator. For 'hat' and 'tilde' operators,
    the eligible indices are the complement of the field's indices.

    Args:
        ops (list): A list of field operators.

    Returns:
        IndexSet: An IndexSet object representing the eligible indices.

    Note:
        This function assumes that all operators have index dimension is 1.
    """

    # Make sure that all ops have the correct index dimension
    for op in ops:
        assert op.index_dim == index_dim, f'Operator {op.name} has index dimension {op.index_dim}; expected {index_dim}'

    # Initialize eligible indices to all indices
    eligible_indices = IndexSet(elements=[], is_complement=True, index_dim=index_dim)

    # Intersect eligible indices with those of each field op
    for op in ops:

        # Compute eligible indices for op
        is_complement = op.op_type in ['hat', 'tilde']
        eligible_indices &= IndexSet(elements=op.field.indices, 
                                        is_complement=is_complement, 
                                        index_dim=index_dim)
    
    return eligible_indices


class ParticleRule(Rule2):
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        name: (str)
        rate: (float)
        spec_string: (str)

        Example: 
        spec_str = 'A_hat_i a_tilde_i b_tilde_i'
        """

        # Call superclass constructor with field ops, etc
        super().__init__(*args, **kwargs)

        # Set no index constraint
        self.index_constraint = None

        # Check that index dimension is 1
        assert self.index_dim == 1, 'Particle rule must have a single index'
        

    # Compute and sample eligible indices
    def compute_eligible_indices(self):

        # Compute eligible indices
        self.eligible_indices = _compute_eligible_indices(ops=self.field_ops, index_dim=1)

        # Get eligible index
        self.eligible_index = self.eligible_indices.sample_index()

        # Get number of eligible indices
        if self.eligible_index is None:
            self.num_eligible_indices = 0
        else:
            self.num_eligible_indices = self.eligible_indices.get_set_size()

        # Get eligible rate
        if self.num_eligible_indices == np.inf:
            self.eligible_rate = self.rate
        else:
            self.eligible_rate = self.rate * self.num_eligible_indices


class AsymmetricInteractionRule(Rule2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set no index constraint
        self.index_constraint = None

        # Check that index dimension is 2
        assert self.index_dim == 2, 'Interaction rule must have two indices'

        # Check that only one op has index_dim 2
        index_dim_2_ops = [op for op in self.field_ops if op.index_dim == 2]
        assert len(index_dim_2_ops) == 1, \
            f'Interaction rule must have exactly one operator with two indices; {[op.name for op in index_dim_2_ops]} all have index dimension 2'    

        # Check that the index_dim 2 op is a hat or check operator
        index_dim_2_op = index_dim_2_ops[0]
        assert index_dim_2_op.op_type in ['hat', 'check'], 'Interaction op must be a hat or check operator'

    def compute_eligible_indices(self):

        # Get pos0 ops and eligible indices
        pos0_ops = [op for op in self.field_ops if op.index_dim == 1 and op.index_spec[0] == 0]
        pos0_ixs = _compute_eligible_indices(ops=pos0_ops, index_dim=1  )

        # Get pos1 ops and eligible indices
        pos1_ops = [op for op in self.field_ops if op.index_dim == 1 and op.index_spec[0] == 1]
        pos1_ixs = _compute_eligible_indices(ops=pos1_ops, index_dim=1)

        # Get pair op and eligible indices
        pair_op = [op for op in self.field_ops if op.index_dim == 2][0]
        pair_ixs = _compute_eligible_indices(ops=[pair_op], index_dim=2)

        # Sample eligible indices if op_type is hat
        if pair_op.op_type == 'hat':
            pos0_index = pos0_ixs.sample_index()
            pos1_index = pos1_ixs.sample_index()
            if pos0_index is not None and pos1_index is not None:
                self.eligible_index = (pos0_index[0], pos1_index[0])
                self.num_eligible_indices = pos0_ixs.get_set_size() * pos1_ixs.get_set_size()
            else:
                self.eligible_index = None
                self.num_eligible_indices = 0

        # Sample eligible indices if op_type is check
        elif pair_op.op_type == 'check':

            # Get eligible pairs of indices
            pair_ixs_set = set(p for p in pair_ixs.elements if (p[0],) in pos0_ixs and (p[1],) in pos1_ixs)

            # Sample eligible index if there are any
            if len(pair_ixs_set) > 0:
                self.eligible_index = random.choice(list(pair_ixs_set))
                self.num_eligible_indices = len(pair_ixs_set)   

            # Otherwise no eligible indices
            else:
                self.eligible_index = None
                self.num_eligible_indices = 0

        # Get eligible rate
        if self.num_eligible_indices == np.inf:
            self.eligible_rate = self.rate
        else:
            self.eligible_rate = self.rate * self.num_eligible_indices

        
        
class SymmetricInteractionRule(Rule2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set no index constraint
        self.index_constraint = None

        # Check that index dimension is 2
        assert self.index_dim == 2, 'Interaction rule must have two indices'

        # Check that only one op has index_dim 2
        index_dim_2_ops = [op for op in self.field_ops if op.index_dim == 2]
        assert len(index_dim_2_ops) == 1, \
            f'Interaction rule must have exactly one operator with two indices; {[op.name for op in index_dim_2_ops]} all have index dimension 2'    

        # Check that the index_dim 2 op is a hat or check operator
        index_dim_2_op = index_dim_2_ops[0]
        assert index_dim_2_op.op_type in ['hat', 'check'], 'Interaction op must be a hat or check operator'

    def compute_eligible_indices(self):

        # Get eligible particle indices
        particle_ops = [op for op in self.field_ops if op.index_dim == 1]
        particle_eligible_indices = _compute_eligible_indices(ops=particle_ops, index_dim=1) 

        # Get eligible interaction indices
        pair_op = [op for op in self.field_ops if op.index_dim == 2][0]
        interaction_eligible_indices = _compute_eligible_indices(ops=[pair_op], index_dim=2)

        # Sample eligible indices if op_type is hat
        if pair_op.op_type == 'hat':
            two_particle_indices = particle_eligible_indices.sample_indices(num_indices=2)
            if two_particle_indices is not None:
                ix1, ix2 = two_particle_indices
                self.eligible_index = (ix1[0], ix2[0]) if ix1[0] < ix2[0] else (ix2[0], ix1[0])
                n = particle_eligible_indices.get_set_size()
                self.num_eligible_indices = n * (n-1) / 2
            else:
                self.eligible_index = None
                self.num_eligible_indices = 0

        # # Sample eligible indices if op_type is check
        # elif pair_op.op_type == 'check':
        #     self.eligible_index = interaction_eligible_indices.sample_index()
        #     self.num_eligible_indices = interaction_eligible_indices.get_set_size()

        # Sample eligible indices if op_type is check
        elif pair_op.op_type == 'check':

            # Get eligible pairs of indices
            pair_ixs_set = set(p for p in interaction_eligible_indices.elements if (p[0],) in particle_eligible_indices and (p[1],) in particle_eligible_indices)

            # Sample eligible index if there are any
            if len(pair_ixs_set) > 0:
                self.eligible_index = random.choice(list(pair_ixs_set))
                self.num_eligible_indices = len(pair_ixs_set)   

            # Otherwise no eligible indices
            else:
                self.eligible_index = None
                self.num_eligible_indices = 0

        # Get eligible rate
        if self.num_eligible_indices == np.inf:
            self.eligible_rate = self.rate
        else:
            self.eligible_rate = self.rate * self.num_eligible_indices
