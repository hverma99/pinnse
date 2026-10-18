Network architectures
=====================

.. py:module:: pinnse.PINNs

.. currentmodule:: pinnse

Feedforward and recurrent network architectures. All are ``torch.nn.Module`` subclasses with a single-tensor forward pass, so any of them can be passed to :class:`~pinnse.Training`.

.. autoclass:: ANN

.. autoclass:: SANN

.. autoclass:: MultiHeadANN

.. autoclass:: Fourier_ANN

.. autoclass:: RecurrentANN
