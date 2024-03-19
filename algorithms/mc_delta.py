from functools import partial
from itertools import product
from multiprocessing import Pool
import numpy as np
from sklearn import neighbors
from tqdm import tqdm, trange

from sqhsrc.utils.exceptions import (
    UnImpException, ParamError, StepWarning
)
from sqhsrc.utils.mc_delta_tools import (
    eval_utility, eval_simi, eval_svc, power_set, get_ele_idxs,
    split_permutations_t_list, split_permutation_num, clock
)

class DynaShap(object):
    """A base class for dynamic Shapley value computation."""

    def __init__(self, x_train, y_train, x_test, y_test,
                 model, init_sv, **kwargs) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = model
        self.init_sv = init_sv

    def add_single_point(self, add_point_x, add_point_y, m=-1, **kwargs
                         ) -> np.ndarray:
        raise UnImpException('add_single_point')

    def del_single_point(self, del_point_idx, m=-1, **kwargs
                         ) -> np.ndarray:
        raise UnImpException('del_single_point')

    def add_multi_points(self, add_points_x, add_points_y, m=-1, **kwargs) -> np.ndarray:
        raise UnImpException('add_multi_points')

    def del_multi_points(self, del_points_idx, m=-1, **kwargs) -> np.ndarray:
        raise UnImpException('del_multi_points')

class DeltaShap(DynaShap):
    """Delta based algorithm for dynamically add/delete a single point.
    """

    def __init__(self, x_train, y_train, x_test, y_test,
                 model, init_sv) -> None:
        super().__init__(x_train, y_train, x_test, y_test,
                         model, init_sv)
        self.m = None

    @clock
    def add_single_point(self, add_point_x, add_point_y, m, **kwargs) -> np.ndarray:
        """
        Add a single point and update the Shapley value with delta based
        algorithm.

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param int m:                   the number of permutations
        :param int num_proc:            (optional) the number of proc. Defaults to 1.
        :param bool cont:               (optional) update init sv for next op.
                                        Defaults to ``False``.
        :return: Shapley value array `delta_sv`
        :rtype: numpy.ndarray
        """

        self.m = m

        num_proc = kwargs.pop('num_proc', 1)
        cont = kwargs.pop('cont', False)

        # assign the permutation of each process
        args = split_permutation_num(m, num_proc)
        pool = Pool()
        func = partial(DeltaShap._delta_add_sub_task,
                       self.x_train, self.y_train,
                       self.x_test, self.y_test,
                       self.model, add_point_x, add_point_y)
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret)
        delta = np.sum(ret_arr, axis=0) / m
        delta_sv = np.append(self.init_sv, 0) + delta

        if cont:
            self.x_train = np.append(self.x_train, [add_point_x], axis=0)
            self.y_train = np.append(self.y_train, add_point_y)
            self.init_sv = delta_sv

        return delta_sv

    @staticmethod
    def _delta_add_sub_task(x_train, y_train, x_test, y_test, model,
                            add_point_x, add_point_y, local_m) -> np.ndarray:
        local_state = np.random.RandomState(None)

        n = len(y_train)
        idxes = np.arange(n)
        delta = np.zeros(n + 1)

        origin_margin = eval_utility([add_point_x], [add_point_y],
                                     x_test, y_test, model)

        delta[-1] += origin_margin / (n + 1) * local_m

        for _ in trange(local_m):
            local_state.shuffle(idxes)
            for i in range(1, n + 1):
                temp_x, temp_y = (x_train[idxes[:i]],
                                  y_train[idxes[:i]])

                u_no_np = eval_utility(temp_x, temp_y, x_test, y_test,
                                       model)

                # Add the new point
                u_with_np = eval_utility(np.append(temp_x, [add_point_x],
                                                   axis=0),
                                         np.append(temp_y, add_point_y),
                                         x_test, y_test, model)

                current_margin = u_with_np - u_no_np

                delta[idxes[i - 1]] += ((current_margin - origin_margin)
                                        / (n + 1) * i)

                delta[-1] += current_margin / (n + 1)
                origin_margin = current_margin
        return delta

    @clock
    def add_multi_points(self, add_points_x, add_points_y, m=-1, **kwargs) -> np.ndarray:
        # add multi points in one batch -> del + utility sampling
        self.m = m
        num_proc = kwargs.get('num_proc', 1)

        # assign the permutation of each process
        args = split_permutation_num(m, num_proc)
        pool = Pool()
        func = partial(DeltaShap._delta_multi_add_sub_task,
                       self.x_train, self.y_train,
                       self.x_test, self.y_test,
                       self.model, add_points_x, add_points_y)
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        # ret_arr = np.asarray(ret)

        n = len(self.y_train)
        m = len(add_points_y)
        cnt1 = np.zeros((m, n + m))  # counter for Part 1
        cnt2 = np.zeros((m, n + m))  # counter for Part 2
        cnt3 = np.zeros(n)
        new_u1 = np.zeros((m, n + m))
        new_u2 = np.zeros((m, n + m))
        new_u3 = np.zeros(n)
        delta = np.zeros(n + m)

        delta_sv = np.zeros(n + m)
        au = kwargs.get('avg_u', np.zeros(n))
        cnt = kwargs.get('old_m', 1000)

        new_u3 += au * cnt
        cnt3 += cnt

        for (r1, r2, r3, r4, r5, r6, r7) in ret:
            delta += r1
            new_u1 += r2
            new_u2 += r3
            new_u3 += r4
            cnt1 += r5
            cnt2 += r6
            cnt3 += r7

        delta /= self.m
        delta_sv[:n] = self.init_sv + delta[:n]
        # new point

        for i in range(1, m+1):
            total = 1
            coef2 = 1
            # print(cnt1)
            # print(cnt1[i-1][n+m-1])
            # print(new_u1[i-1][n+m-1])
            positive_u = []
            negtive_u = []
            cnt1[cnt1 == 0] = 1
            delta_sv[n+i-1] = np.average(new_u1[i-1] / cnt1[i-1])  # part 1
            positive_u = new_u1[i-1] / cnt1[i-1]
            for j in range(1, n+m):
                if j <= n:
                    # part 2 and part 3
                    total *= (n + m - j) / j
                    coef2 *= (n + 1 - j) / j
                    coef3 = total - coef2
                    tmp_cnt2 = cnt2[i-1][j-1] if cnt2[i-1][j-1] > 0 else 1
                    tmp_cnt3 = cnt3[j - 1] if cnt3[j - 1] > 0 else 1
                    minus_sv = (new_u2[i-1][j-1] / tmp_cnt2) * coef3 / total + \
                               (new_u3[j - 1] / tmp_cnt3) * coef2 / total
                    delta_sv[n + i - 1] -= 1 / (n+m) * minus_sv
                    negtive_u.append(minus_sv)
                else:
                    tmp_cnt2 = cnt2[i - 1][j -
                                           1] if cnt2[i - 1][j - 1] > 0 else 1
                    minus_sv = (new_u2[i - 1][j - 1] / tmp_cnt2)
                    delta_sv[n + i - 1] -= 1 / (n+m) * minus_sv
                    negtive_u.append(minus_sv)
            print(f'pos u {positive_u} \t, neg u {negtive_u}')
            # delta_sv[n+i-1] -= np.average(new_u1[i-1] / cnt1[i-1])
        return delta_sv

    @staticmethod
    def _delta_multi_add_sub_task(x_train, y_train, x_test, y_test,
                                  model, add_points_x, add_points_y, local_m):
        local_state = np.random.RandomState(None)
        n = len(y_train)
        m = len(add_points_y)
        cnt1 = np.zeros((m, n + m))  # counter for Part 1
        cnt2 = np.zeros((m, n + m))  # counter for Part 2
        cnt3 = np.zeros(n)
        new_u1 = np.zeros((m, n + m))
        new_u2 = np.zeros((m, n + m))
        new_u3 = np.zeros(n)
        idxes = np.arange(n + m)
        delta = np.zeros(n + m)

        new_x_train = np.append(x_train, add_points_x, axis=0)
        new_y_train = np.append(y_train, add_points_y)

        for _ in trange(local_m):
            # Gen an order over [0, ..., n+m-1]
            local_state.shuffle(idxes)
            p = 0
            for j in range(1, n+m+1):
                # Length p no new point
                if idxes[j-1] < n:
                    p = j
                    continue
                break

            origin_margin = None
            # Start from the new point
            for j in range(p+1, n+m+1):
                coal = idxes[:j]
                coal_ex_new_points = list(set(coal) - set(np.arange(n, n+m)))
                u = eval_utility(new_x_train[coal], new_y_train[coal],
                                 x_test, y_test, model)
                if len(coal_ex_new_points) == 0:
                    u_ex_new_points = 0
                else:
                    u_ex_new_points = eval_utility(new_x_train[coal_ex_new_points],
                                                   new_y_train[coal_ex_new_points],
                                                   x_test, y_test, model)
                # the new points in coal
                new_in = set(coal) - set(coal_ex_new_points)
                # the new points not in coal
                new_ex = set(np.arange(n, n+m)) - new_in

                # For new points
                for i in new_in:
                    new_u1[i-n][j-1] += u
                    cnt1[i-n][j-1] += 1
                for i in new_ex:
                    new_u2[i-n][j-1] += u
                    cnt2[i-n][j-1] += 1
                new_u3[len(coal_ex_new_points)-1] += u_ex_new_points
                cnt3[len(coal_ex_new_points)-1] += 1

                # For changes
                if origin_margin is None:
                    origin_margin = u - u_ex_new_points
                else:
                    current_margin = u - u_ex_new_points
                    delta[idxes[j-1]] += current_margin - origin_margin
                    origin_margin = current_margin
        return delta, new_u1, new_u2, new_u3, cnt1, cnt2, cnt3

    @clock
    def del_single_point(self, del_point_idx, m, **kwargs) -> np.ndarray:
        """
        Delete a single point and update the Shapley value with
        delta based algorithm. (KNN & KNN+)

        :param int del_point_idx:   the index of the deleting point
        :param m:                   the number of permutations
        :param proc_num:            the number of proc
        :param dict flags:          (optional) {'flag_update': True or False},
                                    Defaults to ``False``.
        :param dict params:         (unused yet)
        :return: Shapley value array `delta_sv`
        :rtype: numpy.ndarray
        """

        self.m = m
        num_proc = kwargs.pop('num_proc', 1)
        cont = kwargs.pop('cont', False)
        if num_proc <= 0:
            raise ValueError('Invalid proc num.')

        # assign the permutation of each process
        args = split_permutation_num(m, num_proc)
        pool = Pool()
        func = partial(DeltaShap._delta_del_sub_task,
                       self.x_train, self.y_train,
                       self.x_test, self.y_test,
                       self.model, del_point_idx)
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret)
        delta = np.sum(ret_arr, axis=0) / m
        delta_sv = np.delete(self.init_sv, del_point_idx) + delta

        if cont:
            self.x_train = np.delete(self.x_train, del_point_idx, axis=0)
            self.y_train = np.delete(self.y_train, del_point_idx)
            self.init_sv = delta_sv

        return delta_sv

    @staticmethod
    def _delta_del_sub_task(x_train, y_train, x_test, y_test,
                            model, del_point_idx, local_m) -> np.ndarray:
        local_state = np.random.RandomState(None)

        n = len(y_train)
        deleted_idxes = np.delete(np.arange(n), del_point_idx)
        fixed_idxes = np.copy(deleted_idxes)
        delta = np.zeros(n - 1)

        origin_margin = eval_utility([x_train[del_point_idx, :]],
                                     [y_train[del_point_idx]],
                                     x_test, y_test, model)

        for _ in trange(local_m):
            local_state.shuffle(deleted_idxes)
            for j in range(1, n):
                temp_x, temp_y = (x_train[deleted_idxes[:j]],
                                  y_train[deleted_idxes[:j]])

                u_no_op = eval_utility(temp_x, temp_y, x_test,
                                       y_test, model)

                # Add the old point
                temp_x, temp_y = (np.append(temp_x, [x_train[del_point_idx]],
                                            axis=0),
                                  np.append(temp_y, y_train[del_point_idx]))

                u_with_op = eval_utility(temp_x, temp_y, x_test,
                                         y_test, model)

                current_margin = u_with_op - u_no_op

                idx = np.where(fixed_idxes == deleted_idxes[j - 1])[0]
                delta[idx] += ((-current_margin + origin_margin)
                               / n * j)
                origin_margin = current_margin
        return delta

    @clock
    def del_multi_points(self, del_point_idxes, m, proc_num=1,
                         flags=None, params=None) -> np.ndarray:
        """
        Delete a single point and update the Shapley value with
        delta based algorithm.

        :param list del_point_idxes:   the index of the deleting point
        :param m:                   the number of permutations
        :param proc_num:            the number of proc
        :param dict flags:          (optional) {'flag_update': True or False},
                                    Defaults to ``False``.
        :param dict params:         (unused yet)
        :return: Shapley value array `delta_sv`
        :rtype: numpy.ndarray
        """

        self.m = m

        if proc_num <= 0:
            raise ValueError('Invalid proc num.')

        if flags is None:
            flags = {'flag_update': False}

        # assign the permutation of each process
        args = split_permutation_num(m, proc_num)
        pool = Pool()
        func = partial(DeltaShap._delta_multi_del_sub_task,
                       self.x_train, self.y_train,
                       self.x_test, self.y_test,
                       self.model, del_point_idxes)
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret)
        delta = np.sum(ret_arr, axis=0) / m
        delta_sv = self.init_sv + delta
        delta_sv = np.delete(delta_sv, del_point_idxes)

        return delta_sv

    @staticmethod
    def _delta_multi_del_sub_task(x_train, y_train, x_test, y_test,
                                  model, del_point_idxes, local_m) -> np.ndarray:
        local_state = np.random.RandomState(None)

        n = len(y_train)
        idxes = np.arange(n)
        delta = np.zeros(n)

        for _ in trange(local_m):
            local_state.shuffle(idxes)
            # skip the unchanged part
            c_pos = 0
            for j in range(0, n):
                if idxes[j] in del_point_idxes:
                    c_pos = j
                    break
            # skip the del points
            begin_pos = c_pos
            for j in range(begin_pos + 1, n):
                if idxes[j] in del_point_idxes:
                    c_pos = j
            # Without point need to be deleted
            u_no_dp = 0
            if c_pos != 0:
                temp_x, temp_y = (x_train[idxes[:c_pos]],
                                  y_train[idxes[:c_pos]])
                u_no_dp = eval_utility(temp_x, temp_y, x_test,
                                       y_test, model)
            temp_x, temp_y = (x_train[idxes[:c_pos + 1]],
                              y_train[idxes[:c_pos + 1]])
            u_with_dp = eval_utility(temp_x, temp_y, x_test,
                                     y_test, model)
            origin_margin = u_with_dp - u_no_dp
            for j in range(c_pos + 1, n + 1):
                temp_x, temp_y = (x_train[idxes[:j]],
                                  y_train[idxes[:j]])

                u_with_dp = eval_utility(temp_x, temp_y, x_test,
                                         y_test, model)
                idxes_without_dp = list(set(idxes[:j]) - set(del_point_idxes))

                temp_x, temp_y = (x_train[idxes_without_dp],
                                  y_train[idxes_without_dp])

                u_no_dp = eval_utility(temp_x, temp_y, x_test, y_test, model)

                current_margin = u_with_dp - u_no_dp
                delta[idxes[j - 1]] += -current_margin + origin_margin
                origin_margin = current_margin
        return delta