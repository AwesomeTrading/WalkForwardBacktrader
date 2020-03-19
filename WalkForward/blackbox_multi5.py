# v1

import sys
import multiprocessing as mp
import numpy as np
import scipy.optimize as op  # needsto be imported before numba. if only numba get simportet, it crashes
import datetime

# import threading

# import psutil
from numba import jit, njit, int64, float64

import pickle
import os.path

import ray

# def cpu_t(*args):
#     p = psutil.#cpu_percent(interval=None, percpu=True)
#     v = sum(p) / 72
#     print(args[0], ":", v)
#     return


# def cpu(s="#cpu count"):
#     pass
#     # thread = threading.Thread(target=cpu_t, args=[s])
#     # thread.setDaemon(True)
#     # thread.start()


# when installing conda statsmodels, i get 100% cpu, but its not really fatser
# TODO make sure the values with numba are the same as without

# TODO numba+ray https://github.com/numba/numba/issues/4256

native = False
fake = True
# roundit = False
# decimal_count = 5

rayit = False


npfakefilename = "multi_7_native_executor_noray.pck"

compare = False
if compare:
    native = False

npr = np.random.rand
npi = np.random.randint


class FakeNumpy:
    filename = ""
    random_array = []
    pos = 0
    first = False

    def __init__(self, filename):
        print("-------------NEW FakeNumpy instance")
        self.filename = filename

    def realrand(self, *args):
        print("--------------------REAL")
        rand = npr(*args)
        self.random_array.append(rand)
        return rand

    def realrand_i(self, *args):
        print("--------------------REAL")
        rand = npi(*args)
        self.random_array.append(rand)
        return rand

    def random(self, *args):
        if self.first:
            rand = self.realrand(*args)
        else:
            rand = self.random_array[self.pos]
            self.pos += 1
        return rand

    def randint(self, *args):
        if self.first:
            rand = self.realrand_i(*args)
        else:
            rand = self.random_array[self.pos]
            self.pos += 1
        return rand

    def save(self, raypos):
        if not os.path.isfile(self.filename + str(raypos)):
            outfile = open(self.filename + str(raypos), "wb")
            pickle.dump(self.random_array, outfile)
            outfile.close()
            return True
        return False

    def load(self, raypos):
        if os.path.isfile(self.filename + str(raypos)):
            print("-------------", self.filename + str(raypos))
            infile = open(self.filename + str(raypos), "rb")
            self.random_array = pickle.load(infile)
            infile.close()
        else:
            self.first = True


if fake:
    fakenp = FakeNumpy(npfakefilename)
    np.random.rand = fakenp.random
    np.random.randint = fakenp.randint


class Ti:

    cum = 0

    def __init__(self):
        pass

    def start(self):
        self.s = datetime.datetime.now()

    def stop(self):
        self.e = datetime.datetime.now() - self.s
        self.e = self.e.total_seconds()
        self.cum = self.cum + self.e
        return self.e

    def reset(self):
        self.cum = 0


t1 = Ti()
t2 = Ti()
t3 = Ti()


def get_default_executor():
    """
    Provide a default executor (a context manager
    returning an object with a map method).

    This is the multiprocessing Pool object () for python3.

    The multiprocessing Pool in python2 does not have an __enter__
    and __exit__ method, this function provides a backport of the python3 Pool
    context manager.

    Returns
    -------
    Pool : executor-like object
        An object with context manager (__enter__, __exit__) and map method.
    """
    # if sys.version_info > (3, 0):
    Pool = mp.Pool
    return Pool
    # else:
    #     from contextlib import contextmanager
    #     from functools import wraps

    #     @wraps(mp.Pool)
    #     @contextmanager
    #     def Pool(*args, **kwargs):
    #         pool = mp.Pool(*args, **kwargs)
    #         yield pool
    #         pool.terminate()

    #     return Pool


@ray.remote
def search_ray(
    f,
    box,
    n,
    m,
    batch,
    resfile,
    rho0=0.5,
    p=1.0,
    nrand=10000,
    nrand_frac=0.05,
    executor=None,
    sliceid=None,
):
    return search(
        f,
        box,
        n,
        m,
        batch,
        resfile,
        rho0=rho0,
        p=p,
        nrand=nrand,
        nrand_frac=nrand_frac,
        executor=executor,
        sliceid=sliceid,
    )


# def searchit(
#     f,
#     box,
#     n,
#     m,
#     batch,
#     resfile,
#     rho0=0.5,
#     p=1.0,
#     nrand=10000,
#     nrand_frac=0.05,
#     # executor=get_default_executor(),
#     executor=None,
#     sliceid=None,
# ):
#     # space size
#     d = len(box)

#     # adjusting the number of function calls to the batch size
#     print("adjusting the number of function calls to the batch size")
#     if n % batch != 0:
#         n = n - n % batch + batch

#     if m % batch != 0:
#         m = m - m % batch + batch

#     # generating latin hypercube
#     print("generating latin hypercube")
#     points = np.zeros((n, d + 1))
#     points[:, 0:-1] = latin(n, d)

#     search(
#         f=f,
#         box=box,
#         n=n,
#         m=m,
#         batch=batch,
#         resfile=resfile,
#         rho0=rho0,
#         p=p,
#         nrand=nrand,
#         nrand_frac=nrand_frac,
#         executor=executor,
#         sliceid=sliceid,
#         d=d,
#         points=points,
#     )


def search(
    f,
    box,
    n,
    m,
    batch,
    resfile,
    rho0=0.5,
    p=1.0,
    nrand=10000,
    nrand_frac=0.05,
    # executor=get_default_executor(),
    executor=None,
    sliceid=None,
    d=None,
    points=None,
):
    """
    Minimize given expensive black-box function and save results into text file.

    Parameters
    ----------
    f : callable
        The objective function to be minimized.
    box : list of lists
        List of ranges for each parameter.
    n : int
        Number of initial function calls.
    m : int
        Number of subsequent function calls.
    batch : int
        Number of function calls evaluated simultaneously (in parallel).
    resfile : str
        Text file to save results.
    rho0 : float, optional
        Initial "balls density".
    p : float, optional
        Rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower).
    nrand : int, optional
        Number of random samples that is generated for space rescaling.
    nrand_frac : float, optional
        Fraction of nrand that is actually used for space rescaling.
    executor : callable, optional
        Should have a map method and behave as a context manager.
        Allows the user to use various parallelisation tools
        as dask.distributed or pathos.
    """
    if fake:
        fakenp.pos = 0
        fakenp.first = False
        fakenp.load(sliceid)

    print("------------------- BLACKBOX RAYIT:", rayit)
    print("------------------- NATIVE:", native)
    print("------------------- FAKE:", fake)
    # print("------------------- ROUNDIT:", roundit)
    # print("------------------- ROUNDTO:", decimal_count)
    print("------------------- PICKLEFILE:", npfakefilename)
    print("------------------- POSITION:", sliceid)

    print(points)
    # space size
    d = len(box)

    # adjusting the number of function calls to the batch size
    print("adjusting the number of function calls to the batch size")
    if n % batch != 0:
        n = n - n % batch + batch

    if m % batch != 0:
        m = m - m % batch + batch

    # go from normalized values (unit cube) to absolute values (box)
    def cubetobox(x):
        return [box[i][0] + (box[i][1] - box[i][0]) * x[i] for i in range(d)]

    # generating latin hypercube
    print("generating latin hypercube")
    points = np.zeros((n, d + 1))
    t1.start()
    points[:, 0:-1] = latin(n, d)
    t1.stop()
    print("------", t1.cum)
    t1.reset()

    @ray.remote
    def cubetobox_r(ps, fm=None):
        if fm:
            print("fm")
            return f(ps) / fm
        else:
            print("nofm")
            return f(ps)

    @ray.remote
    def cubetobox_r2(fun, ps):
        return list(map(fun, ps))

    t1.start()
    # initial sampling
    print("initial sampling")
    # result_ids = []
    # result_ids2 = []
    # points1 = np.copy(points)
    # points2 = np.copy(points)
    # local1 = []
    # local2 = []

    for i in range(n // batch):
        if not rayit:
            print("with normal")
            with executor() as e:
                points[batch * i : batch * (i + 1), -1] = list(
                    e.map(
                        f,
                        list(map(cubetobox, points[batch * i : batch * (i + 1), 0:-1])),
                    )
                )
        # if not rayit:
        #     # sometimes seems to hang here in the middle
        #     print("with e")
        #     with executor() as e:
        #         points[batch * i : batch * (i + 1), -1] = list(
        #             e.map(
        #                 f,
        #                 list(map(cubetobox, points[batch * i : batch * (i + 1), 0:-1])),
        #             )
        #         )
        if rayit:
            print("with ray")
            result_ids = []
            cubetobox_results = list(
                map(cubetobox, points[batch * i : batch * (i + 1), 0:-1])
            )

            for cr in cubetobox_results:
                result_ids.append(cubetobox_r.remote(cr))

            points[batch * i : batch * (i + 1), -1] = ray.get(result_ids)
    #     if rayit:
    #         result_ids.append(
    #             cubetobox_r2(
    #                 f, list(map(cubetobox, points[batch * i : batch * (i + 1), 0:-1])),
    #             )
    #         )
    # if rayit:
    #     results = ray.get(result_ids)
    #     for i in range(n // batch):
    #         points[batch * i : batch * (i + 1), -1] = results[i]
    t1.stop()
    print("------", t1.cum)
    t1.reset()

    # normalizing function values
    print("normalizing function values")
    fmax = max(abs(points[:, -1]))
    points[:, -1] = points[:, -1] / fmax

    # volume of d-dimensional ball (r = 1)
    if d % 2 == 0:
        v1 = np.pi ** (d / 2) / np.math.factorial(d / 2)
    else:
        v1 = (
            2
            * (4 * np.pi) ** ((d - 1) / 2)
            * np.math.factorial((d - 1) / 2)
            / np.math.factorial(d)
        )

    # subsequent iterations (current subsequent iteration = i*batch+j)
    print("subsequent iterations (current subsequent iteration = i*batch+j)")
    T = np.identity(d)

    for i in range(m // batch):
        # for i in range(2):

        # refining scaling matrix T
        # TODO rayit ? prolly doesnt work with numba
        if d > 1:
            print("refining scaling matrix T")
            fit_noscale = rbf(points, np.identity(d))
            population = np.zeros((nrand, d + 1))
            population[:, 0:-1] = np.random.rand(nrand, d)

            population[:, -1] = list(map(fit_noscale, population[:, 0:-1]))
            # population[:, -1] = fns(fit_noscale, population)

            cloud = population[population[:, -1].argsort()][
                0 : int(nrand * nrand_frac), 0:-1
            ]
            eigval, eigvec = np.linalg.eig(np.cov(np.transpose(cloud)))
            T = [eigvec[:, j] / np.sqrt(eigval[j]) for j in range(d)]
            T = T / np.linalg.norm(T)

        # sampling next batch of points
        print("sampling next batch of points")
        fit = rbf(points, T)
        points = np.append(points, np.zeros((batch, d + 1)), axis=0)

        for j in range(batch):
            r = (
                (rho0 * ((m - 1.0 - (i * batch + j)) / (m - 1.0)) ** p)
                / (v1 * (n + i * batch + j))
            ) ** (1.0 / d)
            cons = [
                {
                    "type": "ineq",
                    "fun": lambda x, localk=k: np.linalg.norm(
                        np.subtract(x, points[localk, 0:-1])
                    )
                    - r,
                }
                for k in range(n + i * batch + j)
            ]
            while True:
                minfit = op.minimize(
                    fit,
                    np.random.rand(d),
                    method="SLSQP",
                    bounds=[[0.0, 1.0]] * d,
                    constraints=cons,
                )
                if np.isnan(minfit.x)[0] == False:
                    break
            points[n + i * batch + j, 0:-1] = np.copy(minfit.x)

        # print(" with executor() as e:")
        # if not rayit:
        #     # seomtimes seems to hang here in the middle
        #     with executor() as e:
        #         print(" with executor() as e:")
        #         points[n + batch * i : n + batch * (i + 1), -1] = (
        #             list(
        #                 e.map(
        #                     f,
        #                     list(
        #                         map(
        #                             cubetobox,
        #                             points[n + batch * i : n + batch * (i + 1), 0:-1],
        #                         )
        #                     ),
        #                 )
        #             )
        #             / fmax
        #         )
        if not rayit:
            print("with normal")
            with executor() as e:
                points[n + batch * i : n + batch * (i + 1), -1] = (
                    list(
                        e.map(
                            f,
                            list(
                                map(
                                    cubetobox,
                                    points[n + batch * i : n + batch * (i + 1), 0:-1],
                                )
                            ),
                        )
                    )
                    / fmax
                )
        if rayit:
            result_ids = []
            print(" with ray:")
            cubetobox_results = list(
                map(cubetobox, points[n + batch * i : n + batch * (i + 1), 0:-1])
            )

            for cr in cubetobox_results:
                result_ids.append(cubetobox_r.remote(cr, fmax))

            points[n + batch * i : n + batch * (i + 1), -1] = ray.get(result_ids)

    # saving results into text file
    print("saving results into text file")
    points[:, 0:-1] = list(map(cubetobox, points[:, 0:-1]))
    points[:, -1] = points[:, -1] * fmax
    points = points[points[:, -1].argsort()]

    labels = [
        " par_" + str(i + 1) + (7 - len(str(i + 1))) * " " + "," for i in range(d)
    ] + [" f_value    "]
    print(points)
    np.savetxt(
        resfile,
        points,
        delimiter=",",
        fmt=" %+1.4e",
        header="".join(labels),
        comments="",
    )

    print("------------------- BLACKBOX RAYIT:", rayit)
    print("------------------- NATIVE:", native)
    print("------------------- FAKE:", fake)
    # print("------------------- ROUNDIT:", roundit)
    # print("------------------- ROUNDTO:", decimal_count)
    print("------------------- PICKLEFILE:", npfakefilename)
    print("------------------- POSITION:", sliceid)
    if fake:
        bg = fakenp.save(sliceid)
        print("------------------- SAVED:", bg)
    return points


# ----------------------------------
# spread function
# @njit()
# def spread_jit(points, n):
#     r = np.float64(0.0)
#     for i in range(n):
#         for j in range(n):
#             if i > j:
#                 r = r + 1.0 / np.linalg.norm(np.subtract(points[i], points[j]))
#     return r


# faster WITHOUT numba
# @jit()
def latin(n, d):
    """
    Build latin hypercube.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Size of space.

    Returns
    -------
    lh : ndarray
        Array of points uniformly placed in d-dimensional unit cube.
    """
    # spread function
    # def spread(points):
    #     return sum(
    #         1.0 / np.linalg.norm(np.subtract(points[i], points[j]))
    #         for i in range(n)
    #         for j in range(n)
    #         if i > j
    #     )

    def spread(points):
        r = sum(
            1.0 / np.linalg.norm(np.subtract(points[i], points[j]))
            for i in range(n)
            for j in range(n)
            if i > j
        )
        if compare:
            print("COMPARE")
            numb = spread_jit(points, n)
            np.testing.assert_almost_equal(numb, r)
        return r

    @njit()
    def spread_jit(points, n):
        r = np.float64(0.0)
        for i in range(n):
            for j in range(n):
                if i > j:
                    r = r + 1.0 / np.linalg.norm(np.subtract(points[i], points[j]))
        return r

    # starting with diagonal shape
    # lh = [[i / (n - 1.0)] * d for i in range(n)]
    lh = np.array([[i / (n - 1.0)] * d for i in range(n)])

    # lh = np.empty((n, d), dtype=np.float64)
    # for i in range(n):
    #     for j in range(d):
    #         lh[i, j] = i / (n - 1.0)

    # minimizing spread function by shuffling
    # minspread = spread(lh, n)
    if native:
        minspread = spread(lh)
    else:
        minspread = spread_jit(lh, n)

    for i in range(1000):
        point1 = np.random.randint(n)
        point2 = np.random.randint(n)
        dim = np.random.randint(d)

        newlh = np.copy(lh)
        newlh[point1, dim], newlh[point2, dim] = newlh[point2, dim], newlh[point1, dim]

        if native:
            newspread = spread(newlh)
        else:
            newspread = spread_jit(newlh, n)

        if newspread < minspread:
            lh = np.copy(newlh)
            minspread = newspread

        i += 1
    return lh


# -----------------
# @njit()
# def phi(points, n, T):
#     r = np.empty((1, n, n), dtype=np.float64)
#     for i in range(n):
#         for j in range(n):
#             p = np.linalg.norm(np.dot(T, np.subtract(points[i, 0:-1], points[j, 0:-1])))
#             r[0, i, j] = p * p * p
#     return r


# TODO test with NUMBA
# @njit()
def rbf(points, T):
    """
    Build RBF-fit for given points (see Holmstrom, 2008 for details) using scaling matrix.

    Parameters
    ----------
    points : ndarray
        Array of multi-d points with corresponding values [[x1, x2, .., xd, val], ...].
    T : ndarray
        Scaling matrix.

    Returns
    -------
    fit : callable
        Function that returns the value of the RBF-fit at a given point.
    """
    print("rbf")
    n = len(points)
    d = len(points[0]) - 1

    def phi(r):
        return r * r * r

    print(1)

    @njit()
    def phi_jit():
        r = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                p = np.linalg.norm(
                    np.dot(T, np.subtract(points[i, 0:-1], points[j, 0:-1]))
                )
                p = p * p * p
                r[i, j] = p
        return r

    # Phi = [
    #     [
    #         phi(
    #             np.linalg.norm(np.dot(T, np.subtract(points[i, 0:-1], points[j, 0:-1])))
    #         )
    #         for j in range(n)
    #     ]
    #     for i in range(n)
    # ]

    if native:
        print("native phi")
        Phi = [
            [
                phi(
                    np.linalg.norm(
                        np.dot(T, np.subtract(points[i, 0:-1], points[j, 0:-1]))
                    )
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
        if compare:
            print("COMPARE")
            numb = phi_jit()
            numb = numb.tolist()
            np.testing.assert_almost_equal(numb, Phi)

    print(2)
    P = np.ones((n, d + 1))
    print(3)
    P[:, 0:-1] = points[:, 0:-1]
    print(4)
    F = points[:, -1]
    print(5)
    M = np.zeros((n + d + 1, n + d + 1))
    print(6)

    if native:
        M[0:n, 0:n] = Phi
    else:
        M[0:n, 0:n] = phi_jit()
    print(7)
    M[0:n, n : n + d + 1] = P
    print(8)
    M[n : n + d + 1, 0:n] = np.transpose(P)
    print(9)
    v = np.zeros(n + d + 1)
    print(10)
    v[0:n] = F
    print(11)
    sol = np.linalg.solve(M, v)
    print(12)
    lam, b, a = sol[0:n], sol[n : n + d], sol[n + d]

    # def fit(x):
    #     return (
    #         sum(
    #             lam[i] * phi(np.linalg.norm(np.dot(T, np.subtract(x, points[i, 0:-1]))))
    #             for i in range(n)
    #         )
    #         + np.dot(b, x)
    #         + a
    #     )

    def fit(x):
        # print("native fit")
        r = (
            sum(
                lam[i] * phi(np.linalg.norm(np.dot(T, np.subtract(x, points[i, 0:-1]))))
                for i in range(n)
            )
            + np.dot(b, x)
            + a
        )
        if compare:
            print("COMPARE")
            numb = fit_jit(x)
            np.testing.assert_almost_equal(numb, r)
        return r

    @njit()
    def fit_jit(x):
        r = np.empty(n, dtype=np.float64)
        for i in range(n):
            kk = np.linalg.norm(np.dot(T, np.subtract(x, points[i, 0:-1])))
            r[i] = lam[i] * (kk * kk * kk)
        r = (
            np.sum(r) + np.dot(b, x) + a
        )  # TODO move out side of function to parallelize?
        return r

    print(13)
    print(14)
    if native:
        return fit
    else:
        return fit_jit
