from typing import Tuple, Union, Sequence

import numpy as np
import torch
from scipy.optimize import differential_evolution
from torchattacks.attack import Attack
import torch.nn.functional as F

DimensionTupleType = Tuple[Union[int, float], Union[int, float]]
DimensionType = Union[DimensionTupleType, Sequence[DimensionTupleType]]


class ScratchThat(Attack):
    def __init__(self, model,
                 population: int = 1,
                 mutation_rate: float = (0.5, 1),
                 crossover_rate: float = 0.7,
                 scratch_type: str = 'line',
                 n_scratches: int = 1,
                 max_iterations: int = 1000,
                 alpha=1,
                 beta=50):

        super().__init__("ScratchThat", model)

        if scratch_type not in ['line', 'bezier']:
            raise ValueError('scratch_type must be line or bezier '
                             '({})'.format(scratch_type))

        if n_scratches <= 0:
            raise ValueError('n_scratches must be >0'
                             '({})'.format(n_scratches))

        self.alpha = alpha
        self.beta = beta
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = population
        self.max_iterations = max_iterations
        self.scratch_type = scratch_type
        self.n_scratches = n_scratches

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        bs, c, h, w = images.shape

        bounds = []

        for _ in range(self.n_scratches):
            #  (x0, y0), (x1, y1)
            bounds.extend([(0, w), (0, h), (0, w), (0, h)])

            if self.scratch_type == 'bezier':
                #  (x2, y2), w
                bounds.extend([(0, w), (0, h), (0, 7)])

            #  RGB
            bounds.extend([(0, 1), (0, 1), (0, 1)])

        print(bounds)

        iteration_stats = []
        adv_images = []

        for idx in range(bs):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]
            source_label = label

            if self.get_mode() != 'default':
                label = self._get_target_label(image, label)

            f, c = self._get_fun(image, label,
                                 target_attack=self._targeted,
                                 source_label=source_label)

            solution = differential_evolution(func=f,
                                              callback=c,
                                              bounds=bounds,
                                              maxiter=self.max_iterations,
                                              popsize=self.population,
                                              mutation=self.mutation_rate,
                                              # init='random',
                                              recombination=self.crossover_rate,
                                              atol=-1,
                                              disp=False,
                                              polish=False)

            iteration_stats.append(solution.nfev)

            adv_image = self._perturb(image, solution.x)
            adv_images.append(adv_image)

        self.required_iterations = iteration_stats

        adv_images = torch.cat(adv_images)
        return adv_images

    def _get_prob(self, image):
        out = self.model(image.to(self.device))
        prob = F.softmax(out, dim=1)
        return prob.detach().cpu().numpy()

    def _get_fun(self, img, label, target_attack=False, source_label=None):
        img = img.to(self.device)

        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        @torch.no_grad()
        def func(solution, solution_as_perturbed=False):
            pert_image = self._perturb(img, solution)

            p = self._get_prob(pert_image)
            p = p[np.arange(len(p)), label]

            if target_attack:
                p = 1 - p

            return p

        @torch.no_grad()
        def callback(solution, convergence, solution_as_perturbed=False):
            pert_image = self._perturb(img, solution)

            p = self._get_prob(pert_image)[0]
            mx = np.argmax(p)

            if target_attack:
                return mx == label
            else:
                return mx != label

        return func, callback

    def _perturb(self, img, solution):

        zeros = torch.zeros_like(img)

        ml = 7 if self.scratch_type == 'line' else 10

        for i in range(self.n_scratches):
            scratch = solution[i * ml: (i + 1) * ml]

            if ml == 7:
                x0, y0, x1, y1, r, g, b = scratch

                rgb = torch.tensor([r, g, b], device=self.device)

                sx, sy = x1 - x0, y1 - y0

                for t in np.linspace(0, 1, 50):
                    px, py = x0 + t * sx, y0 + t * sy
                    zeros[0, :, int(py), int(px)] = rgb
            else:
                x0, y0, x1, y1, x2, y2, w, r, g, b = scratch

                rgb = torch.tensor([r, g, b], device=self.device)

                for t in np.linspace(0, 1, 50):
                    den = (1 - t)**2 + 2 * (1 - t)*t * w + t**2

                    numx = (1 - t)**2 * x0 + 2 * (1 - t)*t * w * x1 + t**2 * x2
                    numy = (1 - t)**2 * y0 + 2 * (1 - t)*t * w * y1 + t**2 * y2

                    x = numx / den
                    y = numy / den

                    zeros[0, :, int(x), int(y)] = rgb

        mask = (zeros == 0).float()

        im = mask * img + (1 - mask) * zeros

        return im
