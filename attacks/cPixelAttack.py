import numpy as np
import torch
from scipy.optimize import differential_evolution
from torchattacks import OnePixel


class cOnePixel(OnePixel):
    def __init__(self, model, pixels=1, steps=75, popsize=400, inf_batch=128):
        super().__init__(model, pixels=pixels,
                         steps=steps,
                         popsize=popsize,
                         inf_batch=inf_batch)

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        batch_size, channel, height, width = images.shape

        bounds = [(0, height), (0, width)]+[(0, 1)]*channel
        bounds = bounds*self.pixels

        popmul = max(1, int(self.popsize/len(bounds)))

        adv_images = []
        iterations = []

        for idx in range(batch_size):
            image, label = images[idx:idx+1], labels[idx:idx+1]

            if self._targeted:
                target_label = target_labels[idx:idx+1]

                def func(delta):
                    return self._loss(image, target_label, delta)

                def callback(delta, convergence):
                    return self._attack_success(image, target_label, delta)

            else:
                def func(delta):
                    return self._loss(image, label, delta)

                def callback(delta, convergence):
                    return self._attack_success(image, label, delta)

            delta = differential_evolution(func=func,
                                           bounds=bounds,
                                           callback=callback,
                                           maxiter=self.steps, popsize=popmul,
                                           init='random',
                                           recombination=1, atol=-1,
                                           polish=False)

            iterations.append(delta.nfev)
            delta = delta.x
            delta = np.split(delta, len(delta)/len(bounds))
            adv_image = self._perturb(image, delta)
            adv_images.append(adv_image)

        self.required_iterations = iterations

        adv_images = torch.cat(adv_images)
        return adv_images
