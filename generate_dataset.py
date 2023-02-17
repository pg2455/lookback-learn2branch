"""
File adapted from https://github.com/pg2455/Hybrid-learn2branch
"""
import os
import argparse
import pickle
import glob
import shutil
import gzip
import math
import numpy as np
import multiprocessing as mp
from pathlib import Path

import pyscipopt as scip
import utilities

class VanillaFullstrongBranchingDataCollector(scip.Branchrule):
    """
    Implements branching policy to be used by SCIP such that data collection required for hybrid models is embedded in it.
    """
    def __init__(self, rng, query_expert_prob=0.60):
        super().__init__()

        self.khalil_root_buffer = {}
        self.obss = []
        self.targets = []
        self.obss_feats = []
        self.exploration_policy = "pscost"
        self.query_expert_prob = query_expert_prob
        self.rng = rng
        self.iteration_counter = 0
        self.state_buffer = {}
        self.raw_obs = {}
        self.feat_names = {}
        self.parent_collected = set()

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):
        self.iteration_counter += 1
        current_node_id = self.model.getCurrentNode().getNumber()
        parent_id = 0 if self.model.getNNodes() == 1 else self.model.getCurrentNode().getParent().getNumber()

        # always collect the root node
        if parent_id in self.parent_collected:
            child_collection = True
            query_expert = True
        else:
            child_collection = False
            query_expert = self.rng.rand() < self.query_expert_prob or self.model.getNNodes() == 1

        if query_expert:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            state = utilities.extract_state(self.model, self.state_buffer)
            state_khalil = utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer)

            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            best_var = cands_[bestcand]

            self.add_obs(best_var, (state, state_khalil), (cands_, scores), child_collection)
            if self.model.getNNodes() == 1:
                self.state = [state, state_khalil, self.obss[0]]

            # add to the parent list only if it was selected randomly
            if not child_collection:
                self.parent_collected.add(current_node_id)

            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED
        else:
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result':result}

    def add_obs(self, best_var, state_, cands_scores=None, child_collection=False):
        """
        Adds sample to the `self.obs` to be processed later at the end of optimization.

        Parameters
        ----------
            best_var : pyscipopt.Variable
                object representing variable in LP
            state_ : tuple
                extracted features of constraints and variables at a node
            cands_scores : np.array
                scores of each of the candidate variable on which expert policy was executed
            child_collection: bool
                whether this node corresponds to the child node

        Return
        ------
        (bool): True if sample is added succesfully. False o.w.
        """
        if self.model.getNNodes() == 1:
            self.obss = []
            self.targets = []
            self.obss_feats = []
            self.map = sorted([x.getCol().getIndex() for x in self.model.getVars(transformed=True)])
            raw_key_obs = 'root'
            parent_number = 0
        else:
            raw_key_obs = 'non_root'
            parent_number = self.model.getCurrentNode().getParent().getNumber()

        cands, scores = cands_scores
        # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
        if any([s < 0 for s in scores]):
            return False

        state, state_khalil = state_
        var_features = state[2]['values']
        cons_features = state[0]['values']
        edge_features = state[1]

        if raw_key_obs not in self.raw_obs:
            obj_norm = np.linalg.norm(state[2]['raw']['coef_raw'])
            self.raw_obs[raw_key_obs] = {'c':state[0]['raw'], 'e':state[1]['raw'], 'v':state[2]['raw'], 'obj_norm': obj_norm}
            self.feat_names = {'c': state[0]['names'], 'v':state[2]['names']}

        # add more features to variables
        cands_index = [x.getCol().getIndex() for x in cands]
        khalil_features = -np.ones((var_features.shape[0], state_khalil.shape[1]))
        cand_ind = np.zeros((var_features.shape[0],1))
        khalil_features[cands_index] = state_khalil
        cand_ind[cands_index] = 1
        var_features = np.concatenate([var_features, khalil_features, cand_ind], axis=1)

        tmp_scores = -np.ones(len(self.map))
        if scores:
            tmp_scores[cands_index] = scores

        self.targets.append(best_var.getCol().getIndex())
        self.obss.append([var_features, cons_features, edge_features])
        depth = self.model.getCurrentNode().getDepth()

        # each element is a node
        self.obss_feats.append({
                        'depth':depth,
                        'scores':np.array(tmp_scores),
                        'iteration': self.iteration_counter,
                        'number': self.model.getCurrentNode().getNumber(),
                        'parent_number': parent_number,
                        'best_var': self.targets[-1],
                        'child_collection': child_collection
                    })

        return True

def make_samples(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """
    while True:
        episode, instance, seed, time_limit, outdir, rng = in_queue.get()

        m = scip.Model()
        m.setIntParam('display/verblevel', 0)
        m.readProblem(f'{instance}')
        utilities.init_scip_params(m, seed=seed)
        m.setIntParam('timing/clocktype', 2)
        m.setRealParam('limits/time', time_limit)
        m.setLongintParam('limits/nodes', node_limit)

        branchrule = VanillaFullstrongBranchingDataCollector(rng, node_record_prob)
        m.includeBranchrule(
            branchrule=branchrule,
            name="Sampling branching rule", desc="",
            priority=666666, maxdepth=-1, maxbounddist=1)

        m.setBoolParam('branching/vanillafullstrong/integralcands', True)
        m.setBoolParam('branching/vanillafullstrong/scoreall', True)
        m.setBoolParam('branching/vanillafullstrong/collectscores', True)
        m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
        m.setBoolParam('branching/vanillafullstrong/idempotent', True)

        out_queue.put({
            "type": 'start',
            "episode": episode,
            "instance": instance,
            "seed": seed
        })

        m.optimize()

        # data storage - root and node data are saved separately.
        # node data carries a reference to the root filename.
        if m.getNNodes() >= 1 and len(branchrule.obss) > 0:
            filenames = []
            max_depth = max(x['depth'] for x in branchrule.obss_feats)
            stats = {'nnodes':m.getNNodes(), 'time':m.getSolvingTime(), 'gap':m.getGap(), 'nobs':len(branchrule.obss)}

            # prepare root data
            sample_state, sample_khalil_state, root_obss = branchrule.state
            sample_cand_scores = branchrule.obss_feats[0]['scores']
            sample_cands = np.where(sample_cand_scores != -1)[0]
            sample_cand_scores = sample_cand_scores[sample_cands]
            cand_choice = np.where(sample_cands == branchrule.targets[0])[0][0]
            root_second_best_largest_score = np.partition(sample_cand_scores, kth=sample_cand_scores.shape[0]-2)[-2] # second largest score is at (len - 2)th index
            root_second_best_sb_vars = np.where(sample_cand_scores == root_second_best_largest_score)[0]

            ### children of root
            children = [x for x in branchrule.obss_feats if x['parent_number'] == 1]
            children_dictionary = prepare_children_dict(children, root_second_best_sb_vars, episode)

            root_filename = f"sample_root_0_{episode}.pkl"
            filenames.append((f"{outdir}/{root_filename}", False)) # filename, children?
            with gzip.open(filenames[-1][0], 'wb') as f:
                branchrule.obss_feats[0]['n_children'] = len(children)
                branchrule.obss_feats[0]['n_siblings'] = 0
                pickle.dump({
                    'type':'root',
                    'episode':episode,
                    'instance': instance,
                    'seed': seed,
                    'stats': stats,
                    'root_state': [sample_state, sample_khalil_state, sample_cands, cand_choice, sample_cand_scores],
                    'obss': [branchrule.obss[0], branchrule.targets[0], branchrule.obss_feats[0], None],
                    'parent_state': None,
                    'sibling_state': None,
                    'children_state': children_dictionary,
                    'max_depth': max_depth,
                    'raw_root_obs': branchrule.raw_obs.get('root', None),
                    'raw_non_root_obs': branchrule.raw_obs.get('non_root', None),
                    'feat_names': branchrule.feat_names,
                    'collected_as_child': False
                }, f)

            # node data
            for i in range(1, len(branchrule.obss)):
                obss_feats = branchrule.obss_feats[i]
                node_filename = get_node_filename(obss_feats, episode)

                # find nearby nodes
                parent = None
                children, siblings = list(), list()
                for _idx, x in enumerate(branchrule.obss_feats):
                    if _idx == i:
                        continue

                    if x['number'] == obss_feats['parent_number']:
                        parent = x

                    if x['parent_number'] == obss_feats['number']:
                        children.append(x)

                    if x['parent_number'] == obss_feats['parent_number']:
                        siblings.append(x)

                #
                obss_feats['n_children'] = len(children)
                obss_feats['n_siblings'] = len(siblings)

                #
                best_sb_vars = np.where(obss_feats['scores']  == np.max(obss_feats['scores']))[0]
                second_best_largest_score = np.partition(obss_feats['scores'], kth=obss_feats['scores'].shape[0]-2)[-2] # second largest score is at (len - 2)th index
                second_best_sb_vars = np.where(obss_feats['scores'] == second_best_largest_score)[0]

                ### parent
                # assert parent is not None, "No parent found in a tree"
                if parent is not None:
                    # cases considered: (a) if second largest is same as the largest (i.e. due to symmetry) (b) if unique second largest is not the same as the largest
                    # (c) if multiple second largest exist which are not the same as the largest (i.e. due to symmetry)
                    parent_filename = get_node_filename(parent, episode) if parent['number'] > 1 else root_filename
                    parent_second_largest_score = np.partition(parent['scores'], kth=parent['scores'].shape[0]-2)[-2] # second largest score is at (len - 2)th index
                    parent_second_largest_index = np.where(parent['scores'] == parent_second_largest_score)[0]

                    # best_sb_vars can't contain parent's sb var as that is already branched upon
                    lookback = any(x in parent_second_largest_index for x in best_sb_vars)
                    parent_dictionary = { 'filename': parent_filename, 'lookback': lookback}

                ### children
                assert len(children) <= 2, f"more than 2 children found in a binary tree: {len(children)} found"
                children_dictionary = prepare_children_dict(children, second_best_sb_vars, episode)

                ### sibling
                assert len(siblings) <= 1, f"more than 1 sibling found in a binary tree: {len(siblings)} found"
                if len(siblings) == 0:
                    sibling_filename = None
                    sibling_lookback = None
                    sibling_same_sb_var = None
                else:
                    sibling = siblings[0]
                    sibling_filename = get_node_filename(sibling, episode)
                    sibling_best_vars = np.where(sibling['scores']  == np.max(sibling['scores']))[0]
                    sibling_lookback = any(x in parent_second_largest_index for x in sibling_best_vars)
                    sibling_same_sb_var = len(set(sibling_best_vars).intersection(set(best_sb_vars))) > 0

                sibling_dictionary = {
                    "filename": sibling_filename,
                    "lookback": sibling_lookback,
                    "same_sb_var": sibling_same_sb_var
                }

                filenames.append((f"{outdir}/{node_filename}", obss_feats['child_collection']))
                with gzip.open(filenames[-1][0], 'wb') as f:
                    pickle.dump({
                        'type' : 'node',
                        'episode':episode,
                        'instance': instance,
                        'seed': seed,
                        'stats': stats,
                        'root_state': root_filename,
                        'parent_state': parent_dictionary,
                        'sibling_state': sibling_dictionary,
                        'children_state': children_dictionary,
                        'obss': [branchrule.obss[i], branchrule.targets[i], branchrule.obss_feats[i], None],
                        'max_depth': max_depth,
                        'collected_as_child': obss_feats['child_collection']
                    }, f)

            out_queue.put({
                "type": "done",
                "episode": episode,
                "instance": instance,
                "seed": seed,
                "filenames":filenames,
                "nnodes":len(filenames),
            })

        m.freeProb()

def send_orders(orders_queue, instances, seed, time_limit, outdir, start_episode):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Input queue from which orders are received.
    instances : list
        list of filepaths of instances which are solved by SCIP to collect data
    seed : int
        initial seed to insitalize random number generator with
    time_limit : int
        maximum time for which to solve an instance while collecting data
    outdir : str
        directory where to save data
    start_episode : int
        episode to resume data collection. It is used if the data collection process was stopped earlier for some reason.
    """
    rng = np.random.RandomState(seed)
    episode = 0
    while True:
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        # already processed; for a broken process; for root dataset to not repeat instances and seed
        if episode <= start_episode:
            episode += 1
            continue

        orders_queue.put([episode, instance, seed, time_limit, outdir, rng])
        episode += 1

def collect_samples(instances, outdir, rng, n_samples, n_jobs, time_limit):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    instances : list
        filepaths of instances which will be solved to collect data
    outdir : str
        directory where to save data
    rng : np.random.RandomState
        random number generator
    n_samples : int
        total number of samples to collect.
    n_jobs : int
        number of CPUs to utilize or number of instances to solve in parallel.
    time_limit : int
        maximum time for which to solve an instance while collecting data
    """
    os.makedirs(outdir, exist_ok=True)

    # start workers
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    # dir to keep samples temporarily; helps keep a prefect count
    tmp_samples_dir = f'{outdir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # if the process breaks due to some reason, resume from this last_episode.
    existing_samples = glob.glob(f"{outdir}/*.pkl")
    last_episode, last_i = -1, 0
    if existing_samples:
        last_episode = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[3]) for x in existing_samples) # episode is 2nd last
        last_i = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[2]) for x in existing_samples) # sample number is the last

    # start dispatcher
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), time_limit, tmp_samples_dir, last_episode),
            daemon=True)
    dispatcher.start()

    i = last_i # for a broken process
    in_buffer = 0
    while i <= n_samples:
        sample = answers_queue.get()

        if sample['type'] == 'start':
            in_buffer += 1

        if sample['type'] == 'done':
            for filename, child_collection in sample['filenames']:
                x = filename.split('/')[-1].split(".pkl")[0]
                os.rename(filename, f"{outdir}/{x}.pkl")

                if child_collection:
                    with open(f"{outdir}/sample_children.txt", "a") as f:
                        f.write(f"{x}.pkl\n")

                i+=1
                print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

            # note this will not result in exactly n_samples
            if  i >= n_samples:
                # early stop dispatcher (hard)
                if dispatcher.is_alive():
                    dispatcher.terminate()
                    print(f"[m {os.getpid()}] dispatcher stopped...")
                break

        if not dispatcher.is_alive():
            break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)

def get_node_filename(obss_feat, episode):
    iteration_counter = obss_feat['iteration']
    parent_number = obss_feat['parent_number']
    depth = obss_feat['depth']
    sb_var = obss_feat['best_var']
    # child_collection = "child" if obss_feat['child_collection'] else "parent"
    child_collection = "node"

    extra_title = f"_P-{parent_number}_D-{depth}_SB-{sb_var}"
    return f"sample_{child_collection}_{iteration_counter}_{episode}{extra_title}.pkl"

def prepare_children_dict(children, second_best, episode):
    children_filenames = list()
    child_best_vars, child_lookback = list(), list()
    for child in children:
        _child_best_vars = np.where(child['scores']  == np.max(child['scores']))[0]
        _child_lookback = any(x in second_best for x in _child_best_vars)
        child_lookback.append(_child_lookback)
        child_best_vars.append(_child_best_vars)
        children_filenames.append(get_node_filename(child, episode))

    children_same_sb_vars = (len(children) == 2) and len(set(child_best_vars[0]).intersection(set(child_best_vars[1]))) > 0
    return {
            "filenames": children_filenames,
            "n_children": len(children),
            "lookback": child_lookback,
            "same_sb_vars": children_same_sb_vars,
            }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # problem parameters
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--node_record_prob',
        help='Probability to use strong branching',
        type=float,
        default=1,
    )
    parser.add_argument(
        '--basedir',
        help="path of the folder where data will be stored",
        type=str,
        default="data/samples"
    )
    parser.add_argument(
        '--train_size',
        help="number of observations in the training dataset",
        type=int,
        default=150000
    )
    parser.add_argument(
        '--valid_size',
        help="number of observations in the validation dataset",
        type=int,
        default=30000
    )
    parser.add_argument(
        '--test_size',
        help="number of observations in the test dataset",
        type=int,
        default=30000
    )
    parser.add_argument(
        '--mediumvalid_size',
        help="number of observations in the validation dataset for medium sized instances (used for indset problems)",
        type=int,
        default=2000
    )
    args = parser.parse_args()

    train_size = args.train_size
    valid_size = args.valid_size
    test_size = args.test_size
    time_limit = 3600
    node_limit = 1000

    node_record_prob = 0.05
    basedir= args.basedir
    Path(basedir).mkdir(exist_ok=True, parents=True)

    with open(f"{basedir}/node_record_prob.txt", "w") as f:
        f.write(str(node_record_prob))

    # get instance filenames
    if args.problem == 'setcover':
        instances_train = glob.glob('data/instances/setcover/train_500r_1000c_0.05/*.lp')
        instances_valid = glob.glob('data/instances/setcover/valid_500r_1000c_0.05/*.lp')
        instances_test = glob.glob('data/instances/setcover/test_500r_1000c_0.05/*.lp')
        out_dir = f'{basedir}/setcover/500r_1000c_0.05'

    elif args.problem == 'cauctions':
        instances_train = glob.glob('data/instances/cauctions/train_100_500/*.lp')
        instances_valid = glob.glob('data/instances/cauctions/valid_100_500/*.lp')
        instances_test = glob.glob('data/instances/cauctions/test_100_500/*.lp')
        out_dir = f'{basedir}/cauctions/100_500'

    elif args.problem == 'indset':
        instances_train = glob.glob('data/instances/indset/train_750_4/*.lp')
        instances_valid = glob.glob('data/instances/indset/valid_750_4/*.lp')
        instances_test = glob.glob('data/instances/indset/test_750_4/*.lp')
        out_dir = f'{basedir}/indset/750_4'

    elif args.problem == 'facilities':
        instances_train = glob.glob('data/instances/facilities/train_100_100_5/*.lp')
        instances_valid = glob.glob('data/instances/facilities/valid_100_100_5/*.lp')
        instances_test = glob.glob('data/instances/facilities/test_100_100_5/*.lp')
        out_dir = f'{basedir}/facilities/100_100_5'
        time_limit = 600

    else:
        raise NotImplementedError

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")

    if train_size:
        rng = np.random.RandomState(args.seed + 1)
        collect_samples(instances_train, out_dir +"/train", rng, train_size, args.njobs, time_limit)
        print(f"{args.problem} Success: Train data collection ")

    if valid_size:
        rng = np.random.RandomState(args.seed + 1)
        collect_samples(instances_valid, out_dir +"/valid", rng, valid_size, args.njobs, time_limit)
        print(f"{args.problem} Success: Valid data collection")

    if test_size:
        rng = np.random.RandomState(args.seed + 1)
        collect_samples(instances_test, out_dir +"/test", rng, test_size, args.njobs, time_limit)
        print(f"{args.problem} Success: Test data collection")

    if args.problem == "indset" and args.mediumvalid_size:
        mediumvalid_size = args.mediumvalid_size
        instances_mediumvalid = glob.glob('data/instances/indset/mediumvalid_1000_4/*.lp')
        out_dir = f'{basedir}/indset/1000_4'

        rng = np.random.RandomState(args.seed + 1)
        collect_samples(instances_mediumvalid, out_dir +"/mediumvalid", rng, mediumvalid_size, args.njobs, time_limit)
        print(f"{args.problem} Success: Medium validation data collection")
