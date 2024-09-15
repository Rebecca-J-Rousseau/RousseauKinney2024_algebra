import numpy as np
import random
import time
from tqdm.notebook import tqdm
from collections.abc import Iterable


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
    def __init__(self):
        self.fields_dict = {}

    def add_field(self, field):
        self.fields_dict[field.name] = field

    def initialize_to_vacuum(self):
        for field in self.fields_dict.values():
            field.indices.set_to_vacuum()

    def report_state(self):
        for field in self.fields:
            print(f'{field.name}: {field}')
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
                field = Field(name=field_name, index_dim=1)
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
        conjugate_rule = Rule(name=name,
                               rate=name,
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


class ParticleRule(Rule):
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

        assert self.index_dim == 1, 'Particle rule must have a single index'
        

    def compute_eligible_indices(self):

        # Hat operators
        hat_ops = [op for op in self.field_ops if op.op_type == 'hat']
        if len(hat_ops) > 0:
            all_hat_indices = set.union(*[set(i[0] for i in op.field.indices) for op in hat_ops])
        else:
            all_hat_indices = set([])

        # Tilde operators
        tilde_ops = [op for op in self.field_ops if op.op_type == 'tilde']
        if len(tilde_ops) > 0:
            all_tilde_indices = set.union(*[set(i[0] for i in op.field.indices) for op in tilde_ops])
        else:
            all_tilde_indices = set([])

        # Bar operators
        bar_ops = [op for op in self.field_ops if op.op_type == 'bar']
        if len(bar_ops) > 0:    
            shared_bar_indices = set.intersection(*[set(i[0] for i in op.field.indices) for op in bar_ops])
        else:
            shared_bar_indices = set([])

        # Check operators
        check_ops = [op for op in self.field_ops if op.op_type == 'check']
        if len(check_ops) > 0:  
            shared_check_indices = set.intersection(*[set(i[0] for i in op.field.indices) for op in check_ops])
        else:
            shared_check_indices = set([])

        # Get conditioning indices
        conditioning_indices = shared_bar_indices | shared_check_indices

        # Get inelligible indices
        inelligible_indices = all_hat_indices | all_tilde_indices

        # If there are no conditioning indices, we can choose an arbitrary 
        # index that is not inelligible
        if len(conditioning_indices)==0:
            if len(inelligible_indices) > 0:
                self.eligible_index = max(inelligible_indices) + 1
            else:
                self.eligible_index = 0
            self.num_eligible_indices = np.inf
            self.eligible_rate = self.rate 

        # Alternative, if there are conditioning indices, we can choose an 
        # index that is in the conditioning set and not inelligible
        else:
            eligible_indices = conditioning_indices - inelligible_indices
            self.eligible_index = random.choice(list(eligible_indices))
            self.num_eligible_indices = len(eligible_indices)
            self.eligible_rate = self.rate * self.num_eligible_indices
            

class InteractionRule(Rule2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set no index constraint
        self.index_constraint = None

        assert self.index_dim == 2, 'Interaction rule must have two indices'
        
    def compute_eligible_indices(self):
        pass