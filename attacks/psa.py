from itertools import chain
from typing import Tuple, Union, Sequence

import numpy as np
import torch
from scipy.optimize import differential_evolution
from torchattacks.attack import Attack
import torch.nn.functional as F
from torchvision.transforms.functional import resize

DimensionTupleType = Tuple[Union[int, float], Union[int, float]]
DimensionType = Union[DimensionTupleType, Sequence[DimensionTupleType]]


class PatchesSwap(Attack):
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
                 swap=True
                 ):

        super().__init__("PatchesSwap", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        if algorithm not in ['de', 'random']:
            raise ValueError('algorithm must be and integer de or random'
                             '({})'.format(algorithm))

        self.population = population

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

    def forward(self, image, label):

        assert len(image.shape) == 3 or \
               (len(image.shape) == 4 and image.size(0) == 1)

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        if self._targeted:
            label = self._get_target_label(image, label)

        f, c = self._get_fun(image, label,
                             target_attack=self._targeted)

        patch1_x_bounds = tuple(
            [d if isinstance(d, int) else round(image.size(3) * d)
             for d in self.p1_x_dimensions])
        patch1_y_bounds = tuple(
            [d if isinstance(d, int) else round(image.size(2) * d)
             for d in self.p1_y_dimensions])

        if self.p2_x_dimensions is not None:
            patch2_x_bounds = tuple(
                [d if isinstance(d, int) else round(image.size(3) * d)
                 for d in self.p2_x_dimensions])
            patch2_y_bounds = tuple(
                [d if isinstance(d, int) else round(image.size(2) * d)
                 for d in self.p2_y_dimensions])

        # patch2_bounds = tuple(
        #     [d if isinstance(d, int) else round(image.size(3) * d)
        #      for d in self.p2_x_dimensions]), \
        #                 tuple([d if isinstance(d, int) else round(
        #                     image.size(3) * d)
        #                        for d in self.p2_y_dimensions])
        iteratios = 0

        for r in range(self.restarts + 1):

            if r > 0 and self.restart_callback and c(image, None, True):
                break

            if self.algorithm == 'de':
                square = [(0, 1), (0, 1)]
                # dx = tuple([d if isinstance(d, int) else round(image.size(3) * d)
                #             for d in self.p1_x_dimensions])
                # dy = tuple([d if isinstance(d, int) else round(image.size(3) * d)
                #             for d in self.p1_y_dimensions])
                #
                # patch = [dx, dy]

                bounds = square + [patch1_x_bounds] + [patch1_y_bounds] + square

                if not self.same_size:
                    # dx = tuple(
                    #     [d if isinstance(d, int) else round(image.size(3) * d)
                    #      for d in self.p2_x_dimensions])
                    # dy = tuple(
                    #     [d if isinstance(d, int) else round(image.size(3) * d)
                    #      for d in self.p2_y_dimensions])

                    bounds += [patch2_x_bounds] + [patch2_y_bounds]

                # bounds = square + square

                pop = max(5, len(bounds) // self.population)
                # pop = self.population
                max_iter = self.max_iterations

                delta = differential_evolution(func=f,
                                               callback=c,
                                               bounds=bounds,
                                               maxiter=max_iter,
                                               popsize=pop,
                                               # init='random',
                                               recombination=0.8,
                                               atol=-1,
                                               disp=False,
                                               polish=False)

                solution = delta.x
                iteratios += delta.nfev

                image = self._perturb(image, solution).to(self.device)

            else:
                best_p = np.inf
                best_image = image.clone()
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

                    xl1, yl1 = np.random.randint(*patch1_x_bounds), \
                               np.random.randint(*patch1_y_bounds)

                    if not self.same_size:
                        # dx = tuple(
                        #     [d if isinstance(d, int) else round(
                        #         image.size(3) * d)
                        #      for d in self.p2_x_dimensions])
                        # dy = tuple(
                        #     [d if isinstance(d, int) else round(
                        #         image.size(3) * d)
                        #      for d in self.p2_y_dimensions])

                        xl2, yl2 = np.random.randint(*patch2_x_bounds), \
                                   np.random.randint(*patch2_y_bounds)

                        solution = [x1, y1, xl1, yl1, x2, y2, xl2, yl2]
                    else:
                        solution = [x1, y1, xl1, yl1, x2, y2]

                    pert_image = self._perturb(image, solution)

                    if c(pert_image, None, True):
                        best_image = pert_image
                        stop = True
                        break

                    p = f(pert_image, True)
                    if p < best_p:
                        best_p = p
                        best_image = pert_image

                image = best_image

                if stop:
                    break

        return image, iteratios

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
