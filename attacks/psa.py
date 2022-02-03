from itertools import chain
from typing import Tuple, Union, Sequence

import numpy as np
import torch
from torchattacks.attack import Attack
import torch.nn.functional as F

DimensionTupleType = Tuple[Union[int, float], Union[int, float]]
DimensionType = Union[DimensionTupleType, Sequence[DimensionTupleType]]


class PixleAttack(Attack):
    def __init__(self, model,
                 x_dimensions: DimensionType = (2, 10),
                 y_dimensions: DimensionType = (2, 10),
                 pixel_mapping='similarity',
                 restarts: int = 0,
                 swap: bool = False,
                 restart_callback: bool = True,
                 max_iterations: int = 100,
                 penalize_epsilon=0.1,
                 update_each_iteration=False, **kwargs):

        super().__init__("Pixle", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        # self.tol = tol
        self.update_each_iteration = update_each_iteration
        self.max_patches = max_iterations

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

        p = self._get_prob(img)
        p = p[np.arange(len(p)), label]

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

            if target_attack:
                p = 1 - p

            return p.sum()

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
            else:
                return mx != label

        return func, callback

    def _perturb(self, source, solution, destination=None):
        if destination is None:
            destination = source

        c, h, w = source.shape[1:]


        x, y, xl, yl = solution[:4]
        destinations = solution[4:]

        source_pixels = np.ix_(range(c),
                               np.arange(y, y + yl),
                               np.arange(x, x + xl))


        indexes = torch.tensor(destinations)
        destination = destination.clone().detach().to(self.device)

        s = source[0][source_pixels].view(3, -1)

        if self.swap:
            d = destination[0, :, indexes[:, 0], indexes[:, 1]].clone()
            destination[0, :, indexes[:, 0], indexes[:, 1]] = s
            destination[0][source_pixels] = d.unsqueeze(-1)
        else:
            destination[0, :, indexes[:, 0], indexes[:, 1]] = s

        return destination