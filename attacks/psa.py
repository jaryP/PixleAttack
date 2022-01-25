from itertools import chain
from typing import Tuple, Union, Sequence

import numpy as np
import torch
# from scipy.optimize import differential_evolution
from scipy.optimize import differential_evolution
from torch import nn
from torchattacks.attack import Attack
import torch.nn.functional as F
from torchvision.transforms.functional import resize

DimensionTupleType = Tuple[Union[int, float], Union[int, float]]
DimensionType = Union[DimensionTupleType, Sequence[DimensionTupleType]]


# class _PatchesSwap(Attack):
#     def __init__(self, model,
#                  population: int = 1,
#                  p1_x_dimensions: DimensionType = (2, 10),
#                  p1_y_dimensions: DimensionType = (2, 10),
#                  same_size: bool = True,
#                  restarts: int = 0,
#                  restart_callback: bool = True,
#                  algorithm: str = 'de',
#                  max_iterations: int = 100,
#                  p2_x_dimensions: DimensionType = None,
#                  p2_y_dimensions: DimensionType = None,
#                  penalize_epsilon=0.1,
#                  swap=True,
#                  update_each_iteration=False,
#                  tol=0.001,
#                  ):
#
#         super().__init__("PatchesSwap", model)
#
#         if restarts < 0 or not isinstance(restarts, int):
#             raise ValueError('restarts must be and integer >= 0 '
#                              '({})'.format(restarts))
#
#         if algorithm not in ['de', 'random']:
#             raise ValueError('algorithm must be and integer de or random'
#                              '({})'.format(algorithm))
#
#         self.population = population
#
#         self.tol = tol
#         self.update_each_iteration = update_each_iteration
#         self.same_size = same_size
#         self.swap = swap
#
#         self.algorithm = algorithm
#         self.max_iterations = max_iterations
#
#         self.penalize_epsilon = penalize_epsilon
#
#         self.restarts = restarts
#         self.restart_callback = restart_callback
#
#         # if not all([isinstance(d, int) or (isinstance(d, float) and 0 <= d <= 1)
#         #             for d in p1_x_dimensions]):
#         #     raise ValueError('p1_x_dimensions must contains integers or'
#         #                      ' floats in [0, 1]'
#         #                      '({})'.format(p1_x_dimensions))
#
#         if not all([isinstance(d, int) or (isinstance(d, float) and 0 <= d <= 1)
#                     for d in chain(p1_y_dimensions, p1_x_dimensions)]):
#             raise ValueError('dimensions of first patch must contains integers'
#                              ' or floats in [0, 1]'
#                              '({})'.format(p1_y_dimensions))
#
#         self.p1_x_dimensions = p1_x_dimensions
#         self.p1_y_dimensions = p1_y_dimensions
#
#         if not same_size:
#             if p2_y_dimensions is None:
#                 p2_y_dimensions = p1_y_dimensions
#
#             if p2_x_dimensions is None:
#                 p2_x_dimensions = p1_x_dimensions
#
#             if not all([isinstance(d, int) or (
#                     isinstance(d, float) and 0 <= d <= 1)
#                         for d in chain(p2_y_dimensions, p2_x_dimensions)]):
#                 raise ValueError(
#                     'dimensions of second patch must contains integers'
#                     ' or floats in [0, 1]'
#                     '({})'.format(p1_y_dimensions))
#
#             self.p2_x_dimensions = p2_x_dimensions
#             self.p2_y_dimensions = p2_y_dimensions
#         else:
#             self.p2_x_dimensions = None
#             self.p2_y_dimensions = None
#
#         self._supported_mode = ['default', 'targeted']
#
#     def _forward(self, image, label):
#
#         assert len(image.shape) == 3 or \
#                (len(image.shape) == 4 and image.size(0) == 1)
#
#         if len(image.shape) == 3:
#             image = image.unsqueeze(0)
#
#         if self._targeted:
#             label = self._get_target_label(image, label)
#
#         f, c = self._get_fun(image, label,
#                              target_attack=self._targeted)
#
#         patch1_x_bounds = tuple(
#             [d if isinstance(d, int) else round(image.size(3) * d)
#              for d in self.p1_x_dimensions])
#         patch1_y_bounds = tuple(
#             [d if isinstance(d, int) else round(image.size(2) * d)
#              for d in self.p1_y_dimensions])
#
#         if self.p2_x_dimensions is not None:
#             patch2_x_bounds = tuple(
#                 [d if isinstance(d, int) else round(image.size(3) * d)
#                  for d in self.p2_x_dimensions])
#             patch2_y_bounds = tuple(
#                 [d if isinstance(d, int) else round(image.size(2) * d)
#                  for d in self.p2_y_dimensions])
#
#         # patch2_bounds = tuple(
#         #     [d if isinstance(d, int) else round(image.size(3) * d)
#         #      for d in self.p2_x_dimensions]), \
#         #                 tuple([d if isinstance(d, int) else round(
#         #                     image.size(3) * d)
#         #                        for d in self.p2_y_dimensions])
#         iteratios = 0
#
#         statistics = []
#
#         for r in range(self.restarts + 1):
#
#             restart_statistics = [f(image, True)]
#
#             if r > 0 and self.restart_callback and c(image, None, True):
#                 break
#
#             if self.algorithm == 'de':
#                 square = [(0, 1), (0, 1)]
#
#                 bounds = square + [patch1_x_bounds] + [patch1_y_bounds] + square
#
#                 if not self.same_size:
#
#                     bounds += [patch2_x_bounds] + [patch2_y_bounds]
#
#                 # bounds = square + square
#
#                 pop = max(5, len(bounds) // self.population)
#                 # pop = self.population
#                 max_iter = self.max_iterations
#
#                 delta = differential_evolution(func=f,
#                                                callback=c,
#                                                bounds=bounds,
#                                                maxiter=max_iter,
#                                                popsize=pop,
#                                                # init='random',
#                                                recombination=0.8,
#                                                atol=-1,
#                                                disp=False,
#                                                polish=False)
#
#                 solution = delta.x
#                 iteratios += delta.nfev
#
#                 image = self._perturb(image, solution).to(self.device)
#
#                 restart_statistics.append(delta.fun)
#
#             else:
#                 best_p = f(image, True)
#                 best_image = image.clone()
#                 stop = False
#
#                 # iterations_statiss
#
#                 for _ in range(self.max_iterations):
#                     iteratios += 1
#
#                     x1, y1, x2, y2 = np.random.uniform(0, 1, 4)
#                     # dx = tuple(
#                     #     [d if isinstance(d, int) else round(image.size(3) * d)
#                     #      for d in self.p1_x_dimensions])
#                     #
#                     # dy = tuple(
#                     #     [d if isinstance(d, int) else round(image.size(3) * d)
#                     #      for d in self.p1_y_dimensions])
#                     # print(patch1_x_bounds)
#
#                     xl1, yl1 = np.random.randint(patch1_x_bounds[0],
#                                                  patch1_x_bounds[1] + 1), \
#                                np.random.randint(patch1_y_bounds[0],
#                                                  patch1_y_bounds[1] + 1)
#
#                     if not self.same_size:
#                         # dx = tuple(
#                         #     [d if isinstance(d, int) else round(
#                         #         image.size(3) * d)
#                         #      for d in self.p2_x_dimensions])
#                         # dy = tuple(
#                         #     [d if isinstance(d, int) else round(
#                         #         image.size(3) * d)
#                         #      for d in self.p2_y_dimensions])
#
#                         xl2, yl2 = np.random.randint(*patch2_x_bounds), \
#                                    np.random.randint(*patch2_y_bounds)
#
#                         solution = [x1, y1, xl1, yl1, x2, y2, xl2, yl2]
#                     else:
#                         solution = [x1, y1, xl1, yl1, x2, y2]
#
#                     pert_image = self._perturb(image, solution)
#
#                     if c(pert_image, None, True):
#                         best_image = pert_image
#                         stop = True
#                         break
#
#                     p = f(pert_image, True)
#                     # a = best_p * (1 - self.tol)
#                     # print(p, a)
#
#                     if p < (best_p):
#                         # print(p, best_p)
#                         best_p = p
#                         best_image = pert_image
#
#                         restart_statistics.append(best_p)
#
#                         if self.update_each_iteration:
#                             # stop = True
#                             image = pert_image
#                             # break
#
#                         # break
#
#                 image = best_image
#                 statistics.append(restart_statistics)
#
#                 if stop:
#                     break
#
#         return image, iteratios, statistics
#
#     def forward(self, image, label):
#
#         assert len(image.shape) == 3 or \
#                (len(image.shape) == 4 and image.size(0) == 1)
#
#         if len(image.shape) == 3:
#             image = image.unsqueeze(0)
#
#         if self._targeted:
#             label = self._get_target_label(image, label)
#
#         f, c = self._get_fun(image, label,
#                              target_attack=self._targeted)
#
#         patch1_x_bounds = tuple(
#             [d if isinstance(d, int) else round(image.size(3) * d)
#              for d in self.p1_x_dimensions])
#         patch1_y_bounds = tuple(
#             [d if isinstance(d, int) else round(image.size(2) * d)
#              for d in self.p1_y_dimensions])
#
#         if self.p2_x_dimensions is not None:
#             patch2_x_bounds = tuple(
#                 [d if isinstance(d, int) else round(image.size(3) * d)
#                  for d in self.p2_x_dimensions])
#             patch2_y_bounds = tuple(
#                 [d if isinstance(d, int) else round(image.size(2) * d)
#                  for d in self.p2_y_dimensions])
#
#         # patch2_bounds = tuple(
#         #     [d if isinstance(d, int) else round(image.size(3) * d)
#         #      for d in self.p2_x_dimensions]), \
#         #                 tuple([d if isinstance(d, int) else round(
#         #                     image.size(3) * d)
#         #                        for d in self.p2_y_dimensions])
#         iteratios = 0
#
#         statistics = []
#
#         if self.algorithm == 'de':
#             square = [(0, 1), (0, 1)]
#
#             bounds = square + [patch1_x_bounds] + [patch1_y_bounds] + square
#             pop = max(5, len(bounds) // self.population)
#             # pop = self.population
#             max_iter = self.max_iterations
#
#             if not self.same_size:
#                 bounds += [patch2_x_bounds] + [patch2_y_bounds]
#
#             for r in range(self.restarts + 1):
#
#                 if r > 0 and self.restart_callback and c(image, None, True):
#                     break
#
#                 restart_statistics = [f(image, True)]
#
#                 delta = differential_evolution(func=f,
#                                                callback=c,
#                                                bounds=bounds,
#                                                maxiter=max_iter,
#                                                popsize=pop,
#                                                # init='random',
#                                                recombination=0.8,
#                                                atol=-1,
#                                                disp=False,
#                                                polish=False)
#
#                 solution = delta.x
#                 iteratios += delta.nfev
#
#                 image = self._perturb(image, solution).to(self.device)
#
#                 restart_statistics.append(delta.fun)
#
#                 statistics.append(restart_statistics)
#
#         else:
#             best_p = f(image, True)
#             best_image = image.clone()
#             stop = False
#
#             # iterations_statiss
#
#             for _ in range(self.max_iterations):
#                 iteratios += 1
#
#                 x1, y1, x2, y2 = np.random.uniform(0, 1, 4)
#                 # dx = tuple(
#                 #     [d if isinstance(d, int) else round(image.size(3) * d)
#                 #      for d in self.p1_x_dimensions])
#                 #
#                 # dy = tuple(
#                 #     [d if isinstance(d, int) else round(image.size(3) * d)
#                 #      for d in self.p1_y_dimensions])
#                 # print(patch1_x_bounds)
#
#                 xl1, yl1 = np.random.randint(patch1_x_bounds[0],
#                                              patch1_x_bounds[1] + 1), \
#                            np.random.randint(patch1_y_bounds[0],
#                                              patch1_y_bounds[1] + 1)
#
#                 if not self.same_size:
#                     # dx = tuple(
#                     #     [d if isinstance(d, int) else round(
#                     #         image.size(3) * d)
#                     #      for d in self.p2_x_dimensions])
#                     # dy = tuple(
#                     #     [d if isinstance(d, int) else round(
#                     #         image.size(3) * d)
#                     #      for d in self.p2_y_dimensions])
#
#                     xl2, yl2 = np.random.randint(*patch2_x_bounds), \
#                                np.random.randint(*patch2_y_bounds)
#
#                     solution = [x1, y1, xl1, yl1, x2, y2, xl2, yl2]
#                 else:
#                     solution = [x1, y1, xl1, yl1, x2, y2]
#
#                 pert_image = self._perturb(image, solution)
#
#                 p = f(pert_image, True)
#
#                 # a = best_p * (1 - self.tol)
#                 # print(p, a)
#
#                 if p < (best_p):
#                     # print(p, best_p)
#                     best_p = p
#                     best_image = pert_image
#                     statistics.append(p)
#
#                     # if self.update_each_iteration:
#                     # stop = True
#                     image = pert_image
#                     # break
#
#                     # break
#
#                 if c(pert_image, None, True):
#                     best_image = pert_image
#                     stop = True
#                     break
#
#             image = best_image
#             # statistics.append(statistics)
#
#             # if stop:
#             #     break
#
#         return image, iteratios, statistics
#
#     def _get_prob(self, image):
#         out = self.model(image.to(self.device))
#         prob = F.softmax(out, dim=1)
#         return prob.detach().cpu().numpy()
#
#     def _get_fun(self, img, label, target_attack=False):
#         img = img.to(self.device)
#
#         if isinstance(label, torch.Tensor):
#             label = label.numpy()
#
#         @torch.no_grad()
#         def func(solution, solution_as_perturbed=False):
#             if not solution_as_perturbed:
#                 pert_image = self._perturb(img, solution)
#             else:
#                 pert_image = solution
#
#             p = self._get_prob(pert_image)
#             p = p[np.arange(len(p)), label]
#
#             # if self.penalize_epsilon:
#             #     p += torch.linalg.norm((pert_image - img).view(-1),
#             #                            float('inf')).item()
#
#             if target_attack:
#                 p = 1 - p
#
#             return p.sum()
#
#         @torch.no_grad()
#         def callback(solution, convergence, solution_as_perturbed=False):
#             if not solution_as_perturbed:
#                 pert_image = self._perturb(img, solution)
#             else:
#                 pert_image = solution
#
#             p = self._get_prob(pert_image)[0]
#             mx = np.argmax(p)
#
#             if target_attack:
#                 return mx == label
#                 # return p[label] > 0.8
#             else:
#                 return mx != label
#                 # return p[label] < 0.1
#
#         return func, callback
#
#     def _perturb(self, img, solution):
#         c, h, w = img.shape[1:]
#
#         def get_indexes(x, y, xl, yl):
#             if yl > 0:
#                 row_list = np.arange(y, y + yl)
#
#                 mxr = max(row_list)
#                 if mxr >= h:
#                     d = mxr - h
#                     row_list -= d + 1
#             else:
#                 row_list = [y]
#
#             if xl > 0:
#                 col_list = np.arange(x, x + xl)
#
#                 mxc = max(col_list)
#                 if mxc >= w:
#                     d = mxc - w
#                     col_list -= d + 1
#             else:
#                 col_list = [x]
#
#             return row_list, col_list, np.ix_(range(c), row_list, col_list)
#
#         if not self.same_size:
#             (x1, y1, xl1, yl1), (x2, y2, xl2, yl2) = np.split(solution, 2)
#         else:
#             x1, y1, xl1, yl1, x2, y2 = solution
#             xl2, yl2 = xl1, yl1
#
#         xl1, yl1 = int(round(xl1)), int(round(yl1))
#         xl2, yl2 = int(round(xl2)), int(round(yl2))
#
#         x1, y1 = int(x1 * (w - 1)), int(y1 * (h - 1))
#         x2, y2 = int(x2 * (w - 1)), int(y2 * (h - 1))
#
#         s1 = get_indexes(x1, y1, xl1, yl1)[-1]
#         s2 = get_indexes(x2, y2, xl2, yl2)[-1]
#
#         p1 = img[0][s1]
#         p2 = img[0][s2]
#
#         img = img.clone().detach().to(self.device)
#
#         if self.swap:
#             if not self.same_size:
#                 p2_shape = p2.shape[1:]
#                 p2 = resize(p2, p1.shape[1:])
#                 p1 = resize(p1, p2_shape)
#
#             img[0][s1] = p2
#             img[0][s2] = p1
#         else:
#             if not self.same_size:
#                 # p2_shape = p2.shape[1:]
#                 p2 = resize(p2, p1.shape[1:])
#                 # p1 = resize(p1, p2_shape)
#
#             img[0][s1] = p2
#             # img[0][s2] = p1
#
#         return img


class _PatchesSwap(Attack):
    def __init__(self, model,
                 population: int = 1,
                 p1_x_dimensions: DimensionType = (2, 10),
                 p1_y_dimensions: DimensionType = (2, 10),
                 same_size: bool = True,
                 restarts: int = 0,
                 restart_callback: bool = True,
                 algorithm: str = 'de',
                 max_iterations: int = 100,
                 p2_x_dimensions: DimensionType = None,
                 p2_y_dimensions: DimensionType = None,
                 penalize_epsilon=0.1,
                 swap=True,
                 update_each_iteration=False):

        super().__init__("PatchesSwap", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        if algorithm not in ['de', 'random']:
            raise ValueError('algorithm must be and integer de or random'
                             '({})'.format(algorithm))

        self.population = population

        # self.tol = tol
        self.update_each_iteration = update_each_iteration
        self.same_size = same_size
        self.swap = swap

        self.algorithm = algorithm
        self.max_iterations = max_iterations

        self.penalize_epsilon = penalize_epsilon

        self.restarts = restarts
        self.restart_callback = restart_callback

        # if not all([isinstance(d, int) or (isinstance(d, float) and 0 <= d <= 1)
        #             for d in p1_x_dimensions]):
        #     raise ValueError('p1_x_dimensions must contains integers or'
        #                      ' floats in [0, 1]'
        #                      '({})'.format(p1_x_dimensions))

        if not all([isinstance(d, int) or (isinstance(d, float) and 0 <= d <= 1)
                    for d in chain(p1_y_dimensions, p1_x_dimensions)]):
            raise ValueError('dimensions of first patch must contains integers'
                             ' or floats in [0, 1]'
                             '({})'.format(p1_y_dimensions))

        self.p1_x_dimensions = p1_x_dimensions
        self.p1_y_dimensions = p1_y_dimensions

        if not same_size:
            if p2_y_dimensions is None:
                p2_y_dimensions = p1_y_dimensions

            if p2_x_dimensions is None:
                p2_x_dimensions = p1_x_dimensions

            if not all([isinstance(d, int) or (
                    isinstance(d, float) and 0 <= d <= 1)
                        for d in chain(p2_y_dimensions, p2_x_dimensions)]):
                raise ValueError(
                    'dimensions of second patch must contains integers'
                    ' or floats in [0, 1]'
                    '({})'.format(p1_y_dimensions))

            self.p2_x_dimensions = p2_x_dimensions
            self.p2_y_dimensions = p2_y_dimensions
        else:
            self.p2_x_dimensions = None
            self.p2_y_dimensions = None

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):

        assert len(images.shape) == 3 or \
               (len(images.shape) == 4 and images.size(0) == 1)

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        if self._targeted:
            labels = self._get_target_label(images, labels)

        f, c = self._get_fun(images, labels,
                             target_attack=self._targeted)

        patch1_x_bounds = tuple(
            [d if isinstance(d, int) else round(images.size(3) * d)
             for d in self.p1_x_dimensions])
        patch1_y_bounds = tuple(
            [d if isinstance(d, int) else round(images.size(2) * d)
             for d in self.p1_y_dimensions])

        if self.p2_x_dimensions is not None:
            patch2_x_bounds = tuple(
                [d if isinstance(d, int) else round(images.size(3) * d)
                 for d in self.p2_x_dimensions])
            patch2_y_bounds = tuple(
                [d if isinstance(d, int) else round(images.size(2) * d)
                 for d in self.p2_y_dimensions])

        # patch2_bounds = tuple(
        #     [d if isinstance(d, int) else round(image.size(3) * d)
        #      for d in self.p2_x_dimensions]), \
        #                 tuple([d if isinstance(d, int) else round(
        #                     image.size(3) * d)
        #                        for d in self.p2_y_dimensions])

        iteratios = 0

        statistics = []
        adv_images = []

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self._targeted:
        #     label = self._get_target_label(images, label)

        bs, _, _, _ = images.shape

        for idx in range(bs):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]
            best_image = image.clone()

            for r in range(self.restarts + 1):

                restart_statistics = [f(image, True)]

                if self.restart_callback and c(image, None, True):
                    break

                best_p = f(image, True)

                stop = False

                for _ in range(self.max_iterations):
                    iteratios += 1

                    x1, y1, x2, y2 = np.random.uniform(0, 1, 4)
                    # dx = tuple(
                    #     [d if isinstance(d, int) else round(image.size(3) * d)
                    #      for d in self.p1_x_dimensions])
                    #
                    # dy = tuple(
                    #     [d if isinstance(d, int) else round(image.size(3) * d)
                    #      for d in self.p1_y_dimensions])
                    # print(patch1_x_bounds)

                    xl1, yl1 = np.random.randint(patch1_x_bounds[0],
                                                 patch1_x_bounds[1] + 1), \
                               np.random.randint(patch1_y_bounds[0],
                                                 patch1_y_bounds[1] + 1)

                    if not self.same_size:
                        xl2, yl2 = np.random.randint(patch2_x_bounds[0],
                                                     patch2_x_bounds[1] + 1), \
                                   np.random.randint(patch2_y_bounds[0],
                                                     patch2_y_bounds[1] + 1)

                        solution = [x1, y1, xl1, yl1, x2, y2, xl2, yl2]
                    else:
                        solution = [x1, y1, xl1, yl1, x2, y2]

                    pert_image = self._perturb(image, np.asarray(solution))

                    p = f(pert_image, True)

                    if p < best_p:
                        best_p = p
                        best_image = pert_image

                        if self.update_each_iteration:
                            image = pert_image

                        # restart_statistics.append(best_p)

                        # if self.update_each_iteration:
                        #     # stop = True
                        #     image = pert_image
                        # break

                        # break

                    restart_statistics.append(best_p)

                    if c(pert_image, None, True):
                        best_image = pert_image
                        stop = True
                        break

                image = best_image
                statistics.append(restart_statistics)

                if stop:
                    break

            adv_images.append(best_image)
            statistics.append(statistics)

        self.iteration_statistics = statistics
        # self.statistics = {'statistcs': statistics,
        #                    'iterations': iteratios}

        adv_images = torch.cat(adv_images)

        return adv_images

    def _get_prob(self, image):
        out = self.model(image.to(self.device))
        prob = F.softmax(out, dim=1)
        return prob.detach().cpu().numpy()

    def _get_fun(self, img, label, target_attack=False):
        img = img.to(self.device)

        if isinstance(label, torch.Tensor):
            label = label.numpy()

        @torch.no_grad()
        def func(solution, solution_as_perturbed=False):
            if not solution_as_perturbed:
                pert_image = self._perturb(img, solution)
            else:
                pert_image = solution

            p = self._get_prob(pert_image)
            p = p[np.arange(len(p)), label]

            # if self.penalize_epsilon:
            #     p += torch.linalg.norm((pert_image - img).view(-1),
            #                            float('inf')).item()

            if target_attack:
                p = 1 - p

            return p.sum()

        @torch.no_grad()
        def callback(solution, convergence, solution_as_perturbed=False):
            if not solution_as_perturbed:
                pert_image = self._perturb(img, solution)
            else:
                pert_image = solution

            p = self._get_prob(pert_image)[0]
            mx = np.argmax(p)

            if target_attack:
                return mx == label
                # return p[label] > 0.8
            else:
                return mx != label
                # return p[label] < 0.1

        return func, callback

    def _perturb(self, img, solution):
        c, h, w = img.shape[1:]

        def get_indexes(x, y, xl, yl):
            if yl > 0:
                row_list = np.arange(y, y + yl)

                mxr = max(row_list)
                if mxr >= h:
                    d = mxr - h
                    row_list -= d + 1
            else:
                row_list = [y]

            if xl > 0:
                col_list = np.arange(x, x + xl)

                mxc = max(col_list)
                if mxc >= w:
                    d = mxc - w
                    col_list -= d + 1
            else:
                col_list = [x]

            return row_list, col_list, np.ix_(range(c), row_list, col_list)

        if not self.same_size:
            (x1, y1, xl1, yl1), (x2, y2, xl2, yl2) = np.split(solution, 2)
        else:
            x1, y1, xl1, yl1, x2, y2 = solution
            xl2, yl2 = xl1, yl1

        xl1, yl1 = int(round(xl1)), int(round(yl1))
        xl2, yl2 = int(round(xl2)), int(round(yl2))

        x1, y1 = int(x1 * (w - 1)), int(y1 * (h - 1))
        x2, y2 = int(x2 * (w - 1)), int(y2 * (h - 1))

        s1 = get_indexes(x1, y1, xl1, yl1)[-1]
        s2 = get_indexes(x2, y2, xl2, yl2)[-1]

        p1 = img[0][s1]
        p2 = img[0][s2]

        img = img.clone().detach().to(self.device)

        if self.swap:
            if not self.same_size:
                p2_shape = p2.shape[1:]
                p2 = resize(p2, p1.shape[1:])
                p1 = resize(p1, p2_shape)

            img[0][s1] = p2
            img[0][s2] = p1
        else:
            if not self.same_size:
                # p2_shape = p2.shape[1:]
                p2 = resize(p2, p1.shape[1:])
                # p1 = resize(p1, p2_shape)

            img[0][s1] = p2
            # img[0][s2] = p1

        return img


class PatchesSwap(Attack):
    def __init__(self, model,
                 population: int = 1,
                 p1_x_dimensions: DimensionType = (2, 10),
                 p1_y_dimensions: DimensionType = (2, 10),
                 same_size: bool = True,
                 restarts: int = 0,
                 restart_callback: bool = True,
                 algorithm: str = 'de',
                 max_patches: int = 100,
                 p2_x_dimensions: DimensionType = None,
                 p2_y_dimensions: DimensionType = None,
                 penalize_epsilon=0.1,
                 swap=True,
                 update_each_iteration=False):

        super().__init__("PatchesSwap", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        if algorithm not in ['de', 'random']:
            raise ValueError('algorithm must be and integer de or random'
                             '({})'.format(algorithm))

        self.population = population

        # self.tol = tol
        self.update_each_iteration = update_each_iteration
        self.same_size = same_size
        self.swap = swap

        self.algorithm = algorithm
        self.max_patches = max_patches

        self.penalize_epsilon = penalize_epsilon

        self.restarts = restarts
        self.restart_callback = restart_callback

        # if not all([isinstance(d, int) or (isinstance(d, float) and 0 <= d <= 1)
        #             for d in p1_x_dimensions]):
        #     raise ValueError('p1_x_dimensions must contains integers or'
        #                      ' floats in [0, 1]'
        #                      '({})'.format(p1_x_dimensions))

        if not all([isinstance(d, int) or (isinstance(d, float) and 0 <= d <= 1)
                    for d in chain(p1_y_dimensions, p1_x_dimensions)]):
            raise ValueError('dimensions of first patch must contains integers'
                             ' or floats in [0, 1]'
                             '({})'.format(p1_y_dimensions))

        self.p1_x_dimensions = p1_x_dimensions
        self.p1_y_dimensions = p1_y_dimensions

        if not same_size:
            if p2_y_dimensions is None:
                p2_y_dimensions = p1_y_dimensions

            if p2_x_dimensions is None:
                p2_x_dimensions = p1_x_dimensions

            if not all([isinstance(d, int) or (
                    isinstance(d, float) and 0 <= d <= 1)
                        for d in chain(p2_y_dimensions, p2_x_dimensions)]):
                raise ValueError(
                    'dimensions of second patch must contains integers'
                    ' or floats in [0, 1]'
                    '({})'.format(p1_y_dimensions))

            self.p2_x_dimensions = p2_x_dimensions
            self.p2_y_dimensions = p2_y_dimensions
        else:
            self.p2_x_dimensions = None
            self.p2_y_dimensions = None

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):

        assert len(images.shape) == 3 or \
               (len(images.shape) == 4 and images.size(0) == 1)

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        if self._targeted:
            labels = self._get_target_label(images, labels)

        patch1_x_bounds = tuple(
            [d if isinstance(d, int) else round(images.size(3) * d)
             for d in self.p1_x_dimensions])

        patch1_y_bounds = tuple(
            [d if isinstance(d, int) else round(images.size(2) * d)
             for d in self.p1_y_dimensions])

        if self.p2_x_dimensions is not None:
            patch2_x_bounds = tuple(
                [d if isinstance(d, int) else round(images.size(3) * d)
                 for d in self.p2_x_dimensions])
            patch2_y_bounds = tuple(
                [d if isinstance(d, int) else round(images.size(2) * d)
                 for d in self.p2_y_dimensions])
        else:
            patch2_x_bounds = None
            patch2_y_bounds = None

        # patch2_bounds = tuple(
        #     [d if isinstance(d, int) else round(image.size(3) * d)
        #      for d in self.p2_x_dimensions]), \
        #                 tuple([d if isinstance(d, int) else round(
        #                     image.size(3) * d)
        #                        for d in self.p2_y_dimensions])

        iterations = []
        statistics = []
        adv_images = []

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self._targeted:
        #     label = self._get_target_label(images, label)

        bs, _, _, _ = images.shape

        for idx in range(bs):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]
            best_image = image.clone()
            pert_image = image.clone()

            # if self.algorithm != 'de':
            f, c = self._get_fun(image, label,
                                 target_attack=self._targeted)
            # else:
            #     f, c = self._get_fun_de(image, label,
            #                             target_attack=self._targeted)

            best_solution = None
            last_solution = None
            best_p = f(image, destination=None, solution_as_perturbed=True)

            image_probs = [f(image,
                             destination=best_image,
                             solution_as_perturbed=True)]

            tot_it = 0
            it = 0

            for r in range(self.restarts):
                stop = False

                if c(best_image, convergence=None, solution_as_perturbed=True):
                    break

                if self.algorithm == 'de':
                    square = [(0, 1), (0, 1)]

                    bounds = square + [patch1_x_bounds] + \
                             [patch1_y_bounds] + square

                    if not self.same_size:
                        bounds += [patch2_x_bounds] + [patch2_y_bounds]

                    # bounds = square + square

                    pop = max(5, len(bounds) // self.population)
                    # pop = self.population
                    max_iter = self.max_patches

                    f, c = self._get_fun(best_image, label,
                                         target_attack=self._targeted)

                    delta = differential_evolution(func=f,
                                                   callback=c,
                                                   bounds=bounds,
                                                   maxiter=max_iter,
                                                   popsize=1,
                                                   # init='random',
                                                   recombination=0.8,
                                                   atol=-1,
                                                   disp=False,
                                                   polish=False)

                    solution = delta.x
                    tot_it += delta.nfev

                    best_image = self._perturb(best_image,
                                               solution,
                                               destination=None)

                else:
                    for it in range(self.max_patches):

                        x1, y1, x2, y2 = np.random.uniform(0, 1, 4)

                        xl1, yl1 = np.random.randint(patch1_x_bounds[0],
                                                     patch1_x_bounds[1] + 1), \
                                   np.random.randint(patch1_y_bounds[0],
                                                     patch1_y_bounds[1] + 1)

                        if patch2_x_bounds is not None \
                                and patch2_y_bounds is not None:
                            xl2, yl2 = np.random.randint(patch2_x_bounds[0],
                                                         patch2_x_bounds[
                                                             1] + 1), \
                                       np.random.randint(patch2_y_bounds[0],
                                                         patch2_y_bounds[1] + 1)

                            solution = [x1, y1, xl1, yl1, x2, y2, xl2, yl2]
                        else:
                            solution = [x1, y1, xl1, yl1, x2, y2]

                        solution = np.asarray(solution)

                        last_solution = solution

                        pert_image = self._perturb(source=image,
                                                   destination=best_image,
                                                   solution=solution)

                        p = f(pert_image, None, True)

                        if p < best_p:
                            best_p = p
                            best_solution = solution
                            # best_image = pert_image

                            if self.update_each_iteration:
                                best_image = pert_image

                        image_probs.append(best_p)

                        if c(pert_image, None, True):
                            best_image = pert_image
                            stop = True
                            break

                    if best_solution is None:
                        best_image = pert_image

                    elif not self.update_each_iteration \
                            and best_solution is not None:
                        best_image = self._perturb(source=image,
                                                   destination=best_image,
                                                   solution=best_solution)

                    tot_it += it + 1

                    if stop:
                        break

            # image = best_image
            statistics.append(image_probs)
            iterations.append(tot_it)

            adv_images.append(best_image)

        self.probs_statistics = statistics
        self.required_iterations = iterations

        # self.statistics = {'statistcs': statistics,
        #                    'iterations': iteratios}

        adv_images = torch.cat(adv_images)

        return adv_images

    def _get_prob(self, image):
        out = self.model(image.to(self.device))
        prob = F.softmax(out, dim=1)
        return prob.detach().cpu().numpy()

    def _get_fun(self, img, label, target_attack=False):
        img = img.to(self.device)

        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        @torch.no_grad()
        def func(solution,
                 destination=None,
                 solution_as_perturbed=False, **kwargs):

            if not solution_as_perturbed:
                pert_image = self._perturb(source=img,
                                           destination=destination,
                                           solution=solution)
            else:
                pert_image = solution

            p = self._get_prob(pert_image)
            p = p[np.arange(len(p)), label]

            # if self.penalize_epsilon:
            #     p += torch.linalg.norm((pert_image - img).view(-1),
            #                            float('inf')).item()

            if target_attack:
                p = 1 - p

            return p.sum()

        @torch.no_grad()
        def callback(solution,
                     convergence,
                     solution_as_perturbed=False,
                     **kwargs):

            if not solution_as_perturbed:
                pert_image = self._perturb(img, solution, None)
            else:
                pert_image = solution

            p = self._get_prob(pert_image)[0]
            mx = np.argmax(p)

            if target_attack:
                return mx == label
                # return p[label] > 0.8
            else:
                return mx != label
                # return p[label] < 0.1

        return func, callback

    def _perturb(self, source, solution, destination=None):
        if destination is None:
            destination = source

        c, h, w = source.shape[1:]

        def get_indexes(x, y, xl, yl):
            if yl > 0:
                row_list = np.arange(y, y + yl)

                mxr = max(row_list)
                if mxr >= h:
                    d = mxr - h
                    row_list -= d + 1
            else:
                row_list = [y]

            if xl > 0:
                col_list = np.arange(x, x + xl)

                mxc = max(col_list)
                if mxc >= w:
                    d = mxc - w
                    col_list -= d + 1
            else:
                col_list = [x]

            return row_list, col_list, np.ix_(range(c), row_list, col_list)

        if not self.same_size:
            (x1, y1, xl1, yl1), (x2, y2, xl2, yl2) = np.split(solution, 2)
        else:
            x1, y1, xl1, yl1, x2, y2 = solution
            xl2, yl2 = xl1, yl1

        xl1, yl1 = int(round(xl1)), int(round(yl1))
        xl2, yl2 = int(round(xl2)), int(round(yl2))

        x1, y1 = int(x1 * (w - 1)), int(y1 * (h - 1))
        x2, y2 = int(x2 * (w - 1)), int(y2 * (h - 1))

        s1 = get_indexes(x1, y1, xl1, yl1)[-1]
        s2 = get_indexes(x2, y2, xl2, yl2)[-1]

        p1 = source[0][s1]
        p2 = source[0][s2]

        destination = destination.clone().detach().to(self.device)

        if self.swap:
            if not self.same_size:
                p2_shape = p2.shape[1:]
                p2 = resize(p2, p1.shape[1:])
                p1 = resize(p1, p2_shape)

            destination[0][s1] = p2
            destination[0][s2] = p1
        else:
            if not self.same_size:
                # p2_shape = p2.shape[1:]
                p2 = resize(p2, p1.shape[1:])
                # p1 = resize(p1, p2_shape)

            destination[0][s1] = p2
            # img[0][s2] = p1

        return destination


class BlackBoxPatchesSwap(Attack):
    def __init__(self, model,
                 x_dimensions: DimensionType = (2, 10),
                 y_dimensions: DimensionType = (2, 10),
                 pixel_mapping='similarity',
                 restarts: int = 0,
                 swap: bool = False,
                 restart_callback: bool = True,
                 max_patches: int = 100,
                 penalize_epsilon=0.1,
                 update_each_iteration=False, **kwargs):

        super().__init__("PatchesSwap", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        # self.tol = tol
        self.update_each_iteration = update_each_iteration
        self.max_patches = max_patches

        self.penalize_epsilon = penalize_epsilon

        self.restarts = restarts
        self.restart_callback = restart_callback
        self.pixel_mapping = pixel_mapping
        self.swap = swap

        if self.pixel_mapping not in ['random', 'similarity',
                                      'similarity_random', 'distance',
                                      'distance_random']:
            raise ValueError('pixel_mapping must be one of [random, similarity,'
                             'similarity_random, distance, distance_random]'
                             ' ({})'.format(self.pixel_mapping))

        if isinstance(y_dimensions, (int, float)):
            y_dimensions = [y_dimensions, y_dimensions]

        if isinstance(x_dimensions, (int, float)):
            x_dimensions = [x_dimensions, x_dimensions]

        if not all([(isinstance(d, (int)) and d > 0)
                    or (isinstance(d, float) and 0 <= d <= 1)
                    for d in chain(y_dimensions, x_dimensions)]):
            raise ValueError('dimensions of first patch must contains integers'
                             ' or floats in [0, 1]'
                             ' ({})'.format(y_dimensions))

        self.p1_x_dimensions = x_dimensions
        self.p1_y_dimensions = y_dimensions

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        if not self.update_each_iteration:
            return self.restart_forward(images, labels)
        else:
            return self.iterative_forward(images, labels)

    def restart_forward(self, images, labels):

        assert len(images.shape) == 3 or \
               (len(images.shape) == 4 and images.size(0) == 1)

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        if self._targeted:
            labels = self._get_target_label(images, labels)

        x_bounds = tuple(
            [max(1, d if isinstance(d, int) else round(images.size(3) * d))
             for d in self.p1_x_dimensions])

        y_bounds = tuple(
            [max(1, d if isinstance(d, int) else round(images.size(2) * d))
             for d in self.p1_y_dimensions])

        iterations = []
        statistics = []
        adv_images = []

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        bs, _, _, _ = images.shape

        for idx in range(bs):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]

            best_image = image.clone()
            pert_image = image.clone()

            loss, callback = self._get_fun(image, label,
                                           target_attack=self._targeted)
            best_solution = None

            best_p = loss(solution=image, solution_as_perturbed=True)
            image_probs = [best_p]

            tot_it = 0
            it = 0

            for r in range(self.restarts):
                stop = False

                for it in range(self.max_patches):

                    (x, y), (x_offset, y_offset) = \
                        self.get_patch_coordinates(image=image,
                                                   x_bounds=x_bounds,
                                                   y_bounds=y_bounds)

                    destinations = self.get_pixel_mapping(image, x, x_offset,
                                                          y, y_offset,
                                                          destination_image=
                                                          best_image)

                    solution = [x, y, x_offset, y_offset] + destinations

                    pert_image = self._perturb(source=image,
                                               destination=best_image,
                                               solution=solution)

                    p = loss(solution=pert_image,
                             solution_as_perturbed=True)

                    if p < best_p:
                        best_p = p
                        best_solution = pert_image

                    image_probs.append(best_p)

                    if callback(pert_image, None, True):
                        best_solution = pert_image
                        stop = True
                        break

                if best_solution is None:
                    best_image = pert_image
                else:
                    best_image = best_solution

                tot_it += it + 1

                if stop:
                    break

            statistics.append(image_probs)
            iterations.append(tot_it)

            adv_images.append(best_image)

        self.probs_statistics = statistics
        self.required_iterations = iterations

        adv_images = torch.cat(adv_images)

        return adv_images

    def iterative_forward(self, images, labels):

        assert len(images.shape) == 3 or \
               (len(images.shape) == 4 and images.size(0) == 1)

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        if self._targeted:
            labels = self._get_target_label(images, labels)

        x_bounds = tuple(
            [max(1, d if isinstance(d, int) else round(images.size(3) * d))
             for d in self.p1_x_dimensions])

        y_bounds = tuple(
            [max(1, d if isinstance(d, int) else round(images.size(2) * d))
             for d in self.p1_y_dimensions])

        iterations = []
        statistics = []
        adv_images = []

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        bs, _, _, _ = images.shape

        for idx in range(bs):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]

            best_image = image.clone()
            pert_image = image.clone()

            loss, callback = self._get_fun(image, label,
                                           target_attack=self._targeted)

            best_solution = None
            last_solution = None

            best_p = loss(solution=image, solution_as_perturbed=True)
            image_probs = [best_p]

            tot_it = 0
            it = 0

            for it in range(self.max_patches):

                (x, y), (x_offset, y_offset) = \
                    self.get_patch_coordinates(image=image,
                                               x_bounds=x_bounds,
                                               y_bounds=y_bounds)

                destinations = self.get_pixel_mapping(image, x, x_offset,
                                                      y, y_offset,
                                                      destination_image=
                                                      best_image)

                solution = [x, y, x_offset, y_offset] + destinations

                last_solution = solution

                pert_image = self._perturb(source=image,
                                           destination=best_image,
                                           solution=solution)

                p = loss(solution=pert_image, solution_as_perturbed=True)
                
                if p < best_p:
                    best_p = p
                    best_image = pert_image

                image_probs.append(best_p)

                if callback(pert_image, None, True):
                    best_image = pert_image
                    stop = True
                    break

            statistics.append(image_probs)
            iterations.append(it)

            # if best_solution is None:
            #     best_image = pert_image
            #
            # elif not self.update_each_iteration \
            #         and best_solution is not None:
            #     best_image = self._perturb(source=image,
            #                                destination=best_image,
            #                                solution=best_solution)
            #
            # tot_it += it + 1
            #
            # if stop:
            #     break

            # image = best_image

            adv_images.append(best_image)

        self.probs_statistics = statistics
        self.required_iterations = iterations

        adv_images = torch.cat(adv_images)

        return adv_images

    def _get_prob(self, image):
        out = self.model(image.to(self.device))
        prob = F.softmax(out, dim=1)
        return prob.detach().cpu().numpy()

    def loss(self, img, label, target_attack=False):
        # if not solution_as_perturbed:
        #     if sum(solution) == 0:
        #         return np.inf
        #
        #     for i, p in enumerate(solution):
        #         p = round(p)
        #         if p == 1:
        #             pert_image = self._perturb(source=img,
        #                                        destination=pert_image,
        #                                        solution=attacks[i])
        # else:
        #     pert_image = solution

        p = self._get_prob(img)
        p = p[np.arange(len(p)), label]

        # zero_norm = torch.linalg.norm((img - img).view(-1),
        #                               ord=0).item() / 3
        # pixels = img.shape[1] * img.shape[2]
        #
        # perc_change_pixels = zero_norm / pixels

        # if self.penalize_epsilon:
        #     p += torch.linalg.norm((pert_image - img).view(-1),
        #                            float('inf')).item()

        if target_attack:
            p = 1 - p

        return p.sum()

    def get_patch_coordinates(self, image, x_bounds, y_bounds):
        c, h, w = image.shape[1:]

        x, y = np.random.uniform(0, 1, 2)

        x_offset = np.random.randint(x_bounds[0],
                                     x_bounds[1] + 1)

        y_offset = np.random.randint(y_bounds[0],
                                     y_bounds[1] + 1)

        x, y = int(x * (w - 1)), int(y * (h - 1))

        if x + x_offset > w:
            x_offset = w - x

        if y + y_offset > h:
            y_offset = h - y

        return (x, y), (x_offset, y_offset)

    def get_pixel_mapping(self, source_image, x, x_offset, y, y_offset,
                          destination_image=None):

        if destination_image is None:
            destination_image = source_image

        destinations = []
        c, h, w = source_image.shape[1:]
        source_image = source_image[0]

        if self.pixel_mapping == 'random':
            for i in range(x_offset):
                for j in range(y_offset):
                    dx, dy = np.random.uniform(0, 1, 2)
                    dx, dy = int(dx * (w - 1)), int(dy * (h - 1))
                    destinations.append([dx, dy])
        else:
            for i in np.arange(y, y + y_offset):
                for j in np.arange(x, x + x_offset):
                    pixel = source_image[:, i: i + 1, j: j + 1]
                    diff = destination_image - pixel
                    diff = diff[0].abs().mean(0).view(-1)

                    if 'similarity' in self.pixel_mapping:
                        diff = 1 / (1 + diff)
                        # skip pixels with same values
                        diff[diff == 1] = 0

                    probs = torch.softmax(diff, 0).cpu().numpy()

                    indexes = np.arange(len(diff))

                    pair = None

                    linear_iter = iter(sorted(zip(indexes, probs),
                                              key=lambda pit: pit[1],
                                              reverse=True))

                    while True:
                        if 'random' in self.pixel_mapping:
                            index = np.random.choice(indexes, p=probs)
                        else:
                            index = next(linear_iter)[0]

                        _y, _x = np.unravel_index(index, (h, w))

                        if _y == i and _x == j:
                            continue

                        pair = (_x, _y)
                        break

                    destinations.append(pair)

        return destinations

    def _get_fun(self, img, label, target_attack=False):
        img = img.to(self.device)

        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        @torch.no_grad()
        def func(solution,
                 destination=None,
                 solution_as_perturbed=False, **kwargs):

            if not solution_as_perturbed:
                pert_image = self._perturb(source=img,
                                           destination=destination,
                                           solution=solution)
            else:
                pert_image = solution

            p = self._get_prob(pert_image)
            p = p[np.arange(len(p)), label]

            zero_norm = torch.linalg.norm((pert_image - img).view(-1),
                                          ord=0).item() / 3
            pixels = img.shape[1] * img.shape[2]

            perc_change_pixels = zero_norm / pixels

            # if self.penalize_epsilon:
            #     p += torch.linalg.norm((pert_image - img).view(-1),
            #                            float('inf')).item()

            if target_attack:
                p = 1 - p

            return p.sum()
            # return p.sum()

        @torch.no_grad()
        def callback(solution,
                     destination=None,
                     solution_as_perturbed=False,
                     **kwargs):

            if not solution_as_perturbed:
                pert_image = self._perturb(source=img,
                                           destination=destination,
                                           solution=solution)
            else:
                pert_image = solution

            p = self._get_prob(pert_image)[0]
            mx = np.argmax(p)

            if target_attack:
                return mx == label
                # return p[label] > 0.8
            else:
                return mx != label
                # return p[label] < 0.1

        return func, callback

    def _perturb(self, source, solution, destination=None):
        if destination is None:
            destination = source

        c, h, w = source.shape[1:]

        # def get_indexes(x, y, xl, yl):
        #     if yl > 0:
        #         row_list = np.arange(y, min(h, y + yl))
        #
        #         # mxr = max(row_list)
        #         # if mxr >= h:
        #         #     d = mxr - h
        #         #     row_list -= d + 1
        #     else:
        #         row_list = [y]
        #
        #     if xl > 0:
        #         col_list = np.arange(x, min(w, x + xl))
        #
        #         # mxc = max(col_list)
        #         # if mxc >= w:
        #         #     d = mxc - w
        #         #     col_list -= d + 1
        #     else:
        #         col_list = [x]
        #
        #     return row_list, col_list, np.ix_(range(c), row_list, col_list)

        x, y, xl, yl = solution[:4]
        destinations = solution[4:]

        # x1, y1 = int(round(x1)), int(round(y1))
        # xl2, yl2 = int(round(xl1)), int(round(yl1))
        # x1, y1 = int(x1 * (w - 1)), int(y1 * (h - 1))
        # x2, y2 = int(x2 * (w - 1)), int(y2 * (h - 1))

        source_pixels = np.ix_(range(c),
                               np.arange(y, y + yl),
                               np.arange(x, x + xl))

        # rows, cols, s1 = get_indexes(x1, y1, xl1, yl1)
        # dest_cols = [x for x, _ in destinations]
        # dest_rows = [y for _, y in destinations]
        # dest_pixels = np.ix_(range(c), dest_rows, dest_cols)
        # s = source[0][source_pixels]
        # s1 = source[0, :, np.arange(y, y + yl), np.arange(x, x + xl)]
        # d = destination[0, :, dest_rows, dest_cols]
        # d = destination[0, :, destinations[:, 1], destinations[:, 0]]
        # destination[0, :, destinations[:, 1], destinations[:, 0]] =
        # source[0][source_pixels].view(3, -1)
        # grid_x, grid_y = torch.meshgrid(torch.tensor(dest_rows),
        #                                 torch.tensor(dest_cols))
        # d = destination[0, :, grid_x, grid_y]
        # s2 = get_indexes(x2, y2, xl2, yl2)[-1]
        # p1 = source[0][s1]
        # p2 = source[0][a]

        indexes = torch.tensor(destinations)
        destination = destination.clone().detach().to(self.device)

        # if self.swap:
        #     if not self.same_size:
        #         p2_shape = p2.shape[1:]
        #         p2 = resize(p2, p1.shape[1:])
        #         p1 = resize(p1, p2_shape)
        #
        #     destination[0][s1] = p2
        #     destination[0][s2] = p1
        # else:
        #     if not self.same_size:
        #         # p2_shape = p2.shape[1:]
        #         p2 = resize(p2, p1.shape[1:])
        #         # p1 = resize(p1, p2_shape)

        # destination[0][dest_pixels] = source[0][source_pixels]
        # img[0][s2] = p1

        s = source[0][source_pixels].view(3, -1)

        if self.swap:
            d = destination[0, :, indexes[:, 0], indexes[:, 1]].clone()
            destination[0, :, indexes[:, 0], indexes[:, 1]] = s
            destination[0][source_pixels] = d.unsqueeze(-1)
        else:
            # s = source[0][source_pixels].view(3, -1)
            destination[0, :, indexes[:, 0], indexes[:, 1]] = s

        return destination


class WhiteBoxPatchesSwap(Attack):
    def __init__(self, model,
                 x_dimensions: DimensionType = (2, 10),
                 y_dimensions: DimensionType = (2, 10),
                 pixel_mapping='similarity',
                 restarts: int = 0,
                 swap: bool = False,
                 restart_callback: bool = True,
                 max_patches: int = 100,
                 penalize_epsilon=0.1,
                 update_each_iteration=False, **kwargs):

        super().__init__("PatchesSwap", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        # self.tol = tol
        self.update_each_iteration = update_each_iteration
        self.max_patches = max_patches

        self.penalize_epsilon = penalize_epsilon

        self.restarts = restarts
        self.restart_callback = restart_callback
        self.pixel_mapping = pixel_mapping
        self.swap = swap

        if self.pixel_mapping not in ['random', 'similarity',
                                      'similarity_random', 'distance',
                                      'distance_random']:
            raise ValueError('pixel_mapping must be one of [random, similarity,'
                             'similarity_random, distance, distance_random]'
                             ' ({})'.format(self.pixel_mapping))

        if isinstance(y_dimensions, int):
            y_dimensions = [y_dimensions, y_dimensions]

        if isinstance(x_dimensions, int):
            x_dimensions = [x_dimensions, x_dimensions]

        if not all([(isinstance(d, int) and d > 0)
                    or (isinstance(d, float) and 0 <= d <= 1)
                    for d in chain(y_dimensions, x_dimensions)]):
            raise ValueError('dimensions of first patch must contains integers'
                             ' or floats in [0, 1]'
                             ' ({})'.format(y_dimensions))

        self.p1_x_dimensions = x_dimensions
        self.p1_y_dimensions = y_dimensions

        self._supported_mode = ['default', 'targeted']

    # def forward(self, images, labels):
    #     if not self.update_each_iteration:
    #         return self.restart_forward(images, labels)
    #     else:
    #         return self.iterative_forward(images, labels)

    def forward(self, images, labels):

        assert len(images.shape) == 3 or \
               (len(images.shape) == 4 and images.size(0) == 1)

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        if self._targeted:
            labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        iterations = []
        statistics = []
        adv_images = []

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        bs, c, h, w = images.shape

        for idx in range(bs):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]

            best_image = image.clone()
            pert_image = image.clone()

            # loss, callback = self._get_fun(image, label,
            #                                target_attack=self._targeted)
            best_solution = None

            # best_p = loss(solution=image, solution_as_perturbed=True)
            # image_probs = [best_p]

            tot_it = 0
            it = 0
            pert_image.requires_grad = True

            for r in range(self.restarts):
                stop = False

                outputs = self.model(pert_image)

                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, label)
                else:
                    cost = loss(outputs, label)

                print(self.loss(pert_image, label))

                grad = torch.autograd.grad(cost, pert_image,
                                           retain_graph=False,
                                           create_graph=False)[0]

                grad = grad[0].abs().mean(0).view(-1).detach().cpu().numpy()
                indexes = np.arange(len(grad))

                p = np.exp(grad) / np.exp(grad).sum()
                best_p = self.loss(pert_image, label)

                for i in range(self.max_patches):
                    best_image = pert_image.clone()
                    i1, i2 = np.random.choice(indexes, 2, False, p)
                    _y1, _x1 = np.unravel_index(i1, (h, w))
                    _y2, _x2 = np.unravel_index(i2, (h, w))

                    p1 = best_image[0, :, _y1, _x1]
                    p2 = best_image[0, :, _y2, _x2]

                    best_image[0, :, _y1, _x1] = p2
                    best_image[0, :, _y2, _x2] = p1

                    if self.loss(best_image, label) < best_p:
                        best_p = self.loss(best_image, label)
                        pert_image = best_image
                        print(best_p)
                        break

                # top_indexes = torch.topk(grad, w * h)[1].detach().cpu().numpy()
                #
                # best_image = pert_image.clone()
                # best_p = self.loss(pert_image, label)
                #
                # for i in range(0, w * h, 1):
                #
                #     _y1, _x1 = np.unravel_index(top_indexes[i], (h, w))
                #     _y2, _x2 = np.unravel_index(top_indexes[i + 1], (h, w))
                #
                #     p1 = best_image[0, :, _y1, _x1]
                #     p2 = best_image[0, :, _y2, _x2]
                #
                #     best_image[0, :, _y1, _x1] = p2
                #     best_image[0, :, _y2, _x2] = p1
                #
                #     if self.loss(best_image, label) < best_p:
                #         best_p = self.loss(best_image, label)
                #         pert_image = best_image
                #         print(best_p)
                #         break

                if self.callback(pert_image, label):
                    print('Iterations', i)
                    adv_images.append(pert_image.detach())
                    break

        #         for it in range(self.max_patches):
        #
        #             (x, y), (x_offset, y_offset) = \
        #                 self.get_patch_coordinates(image=image,
        #                                            x_bounds=x_bounds,
        #                                            y_bounds=y_bounds)
        #
        #             destinations = self.get_pixel_mapping(image, x, x_offset,
        #                                                   y, y_offset,
        #                                                   destination_image=
        #                                                   best_image)
        #
        #             solution = [x, y, x_offset, y_offset] + destinations
        #
        #             pert_image = self._perturb(source=image,
        #                                        destination=best_image,
        #                                        solution=solution)
        #
        #             p = loss(solution=pert_image,
        #                      solution_as_perturbed=True)
        #
        #             if p < best_p:
        #                 best_p = p
        #                 best_solution = pert_image
        #
        #             image_probs.append(best_p)
        #
        #             if callback(pert_image, None, True):
        #                 best_solution = pert_image
        #                 stop = True
        #                 break
        #
        #         if best_solution is None:
        #             best_image = pert_image
        #         else:
        #             best_image = best_solution
        #
        #         tot_it += it + 1
        #
        #         if stop:
        #             break
        #
        #     statistics.append(image_probs)
        #     iterations.append(tot_it)
        #
        #     adv_images.append(best_image)
        #
        # self.probs_statistics = statistics
        # self.required_iterations = iterations

        adv_images = torch.cat(adv_images)

        return adv_images

    def _get_prob(self, image):
        out = self.model(image.to(self.device))
        prob = F.softmax(out, dim=1)
        return prob.detach().cpu().numpy()

    def loss(self, img, label, target_attack=False):
        p = self._get_prob(img)
        p = p[np.arange(len(p)), label]

        if target_attack:
            p = 1 - p

        return p.sum()

    def callback(self, image, label):
        p = self._get_prob(image)[0]
        mx = np.argmax(p)

        if self._targeted:
            return mx == label
        else:
            return mx != label

    def _get_fun(self, img, label, target_attack=False):
        img = img.to(self.device)

        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        @torch.no_grad()
        def func(solution,
                 destination=None,
                 solution_as_perturbed=False, **kwargs):

            if not solution_as_perturbed:
                pert_image = self._perturb(source=img,
                                           destination=destination,
                                           solution=solution)
            else:
                pert_image = solution

            p = self._get_prob(pert_image)
            p = p[np.arange(len(p)), label]

            zero_norm = torch.linalg.norm((pert_image - img).view(-1),
                                          ord=0).item() / 3
            pixels = img.shape[1] * img.shape[2]

            perc_change_pixels = zero_norm / pixels

            # if self.penalize_epsilon:
            #     p += torch.linalg.norm((pert_image - img).view(-1),
            #                            float('inf')).item()

            if target_attack:
                p = 1 - p

            return p.sum()
            # return p.sum()

        @torch.no_grad()
        def callback(solution,
                     destination=None,
                     solution_as_perturbed=False,
                     **kwargs):

            if not solution_as_perturbed:
                pert_image = self._perturb(source=img,
                                           destination=destination,
                                           solution=solution)
            else:
                pert_image = solution

            p = self._get_prob(pert_image)[0]
            mx = np.argmax(p)

            if target_attack:
                return mx == label
                # return p[label] > 0.8
            else:
                return mx != label
                # return p[label] < 0.1

        return func, callback

    def _perturb(self, source, solution, destination=None):
        if destination is None:
            destination = source

        c, h, w = source.shape[1:]

        # def get_indexes(x, y, xl, yl):
        #     if yl > 0:
        #         row_list = np.arange(y, min(h, y + yl))
        #
        #         # mxr = max(row_list)
        #         # if mxr >= h:
        #         #     d = mxr - h
        #         #     row_list -= d + 1
        #     else:
        #         row_list = [y]
        #
        #     if xl > 0:
        #         col_list = np.arange(x, min(w, x + xl))
        #
        #         # mxc = max(col_list)
        #         # if mxc >= w:
        #         #     d = mxc - w
        #         #     col_list -= d + 1
        #     else:
        #         col_list = [x]
        #
        #     return row_list, col_list, np.ix_(range(c), row_list, col_list)

        x, y, xl, yl = solution[:4]
        destinations = solution[4:]

        # x1, y1 = int(round(x1)), int(round(y1))
        # xl2, yl2 = int(round(xl1)), int(round(yl1))
        # x1, y1 = int(x1 * (w - 1)), int(y1 * (h - 1))
        # x2, y2 = int(x2 * (w - 1)), int(y2 * (h - 1))

        source_pixels = np.ix_(range(c),
                               np.arange(y, y + yl),
                               np.arange(x, x + xl))

        # rows, cols, s1 = get_indexes(x1, y1, xl1, yl1)
        # dest_cols = [x for x, _ in destinations]
        # dest_rows = [y for _, y in destinations]
        # dest_pixels = np.ix_(range(c), dest_rows, dest_cols)
        # s = source[0][source_pixels]
        # s1 = source[0, :, np.arange(y, y + yl), np.arange(x, x + xl)]
        # d = destination[0, :, dest_rows, dest_cols]
        # d = destination[0, :, destinations[:, 1], destinations[:, 0]]
        # destination[0, :, destinations[:, 1], destinations[:, 0]] =
        # source[0][source_pixels].view(3, -1)
        # grid_x, grid_y = torch.meshgrid(torch.tensor(dest_rows),
        #                                 torch.tensor(dest_cols))
        # d = destination[0, :, grid_x, grid_y]
        # s2 = get_indexes(x2, y2, xl2, yl2)[-1]
        # p1 = source[0][s1]
        # p2 = source[0][a]

        indexes = torch.tensor(destinations)
        destination = destination.clone().detach().to(self.device)

        # if self.swap:
        #     if not self.same_size:
        #         p2_shape = p2.shape[1:]
        #         p2 = resize(p2, p1.shape[1:])
        #         p1 = resize(p1, p2_shape)
        #
        #     destination[0][s1] = p2
        #     destination[0][s2] = p1
        # else:
        #     if not self.same_size:
        #         # p2_shape = p2.shape[1:]
        #         p2 = resize(p2, p1.shape[1:])
        #         # p1 = resize(p1, p2_shape)

        # destination[0][dest_pixels] = source[0][source_pixels]
        # img[0][s2] = p1

        s = source[0][source_pixels].view(3, -1)

        if self.swap:
            d = destination[0, :, indexes[:, 0], indexes[:, 1]].clone()
            destination[0, :, indexes[:, 0], indexes[:, 1]] = s
            destination[0][source_pixels] = d.unsqueeze(-1)
        else:
            # s = source[0][source_pixels].view(3, -1)
            destination[0, :, indexes[:, 0], indexes[:, 1]] = s

        return destination
