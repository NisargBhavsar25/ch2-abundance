from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import uuid4

import haiku as hk
import jax
import jax.numpy as jnp
import networkx as nx
import rich

from haiku._src import base
from jax.scipy.integrate import trapezoid
from rich.table import Table
from simpleeval import simple_eval
from jaxspec.model.abc import SpectralModel



class CustomSpectralModel(SpectralModel):
	def flux(self, e_low, e_high, energy_flux=False, n_points=2):
		

		# TODO : enable interpolation and integration with more than 2 points for the continuum

		# if n_points == 2:
		# 	energies = jnp.hstack((e_low, e_high[-1]))
		# 	energies_to_integrate = jnp.stack((e_low, e_high))

		# else:
		# 	energies_to_integrate = jnp.linspace(e_low, e_high, n_points)
		# 	energies = energies_to_integrate

		fine_structures_flux = jnp.zeros_like(e_low)
		runtime_modules = {}
		continuum = {}
		
		energies=jnp.ones(e_low)
		continuum_flux=jnp.zeros_like(e_low)
		# Iterate through the graph in topological order and
		# compute the continuum contribution for each component

		for node_id in nx.dag.topological_sort(self.graph):
			node = self.graph.nodes[node_id]

			# Instantiate the haiku modules
			if node and node["type"] == "component":
				runtime_modules[node_id] = node["component"](name=node["name"], **node["kwargs"])
				continuum[node_id] = runtime_modules[node_id].continuum(energies)

			elif node and node["type"] == "operation":
				component_1 = list(self.graph.in_edges(node_id))[0][0]  # noqa: RUF015
				component_2 = list(self.graph.in_edges(node_id))[1][0]
				continuum[node_id] = node["function"](
					continuum[component_1], continuum[component_2]
				)

		if n_points == 2:
			flux_1D = continuum[list(self.graph.in_edges("out"))[0][0]]  # noqa: RUF015
			flux = jnp.stack((flux_1D[:-1], flux_1D[1:]))

		else:
			flux = continuum[list(self.graph.in_edges("out"))[0][0]]  # noqa: RUF015

		# if energy_flux:
		# 	continuum_flux = trapezoid(
		# 		flux * energies_to_integrate**2,
		# 		x=jnp.log(energies_to_integrate),
		# 		axis=0,
		# 	)

		# else:

		# 	continuum_flux = trapezoid(
		# 		flux * energies_to_integrate, x=jnp.log(energies_to_integrate), axis=0
		# 	)

		# Iterate from the root nodes to the output node and
		# compute the fine structure contribution for each component

		root_nodes = [
			node_id
			for node_id, in_degree in self.graph.in_degree(self.graph.nodes)
			if in_degree == 0 and self.graph.nodes[node_id].get("component_type") == "additive"
		]

		for root_node_id in root_nodes:
			path = nx.shortest_path(self.graph, source=root_node_id, target="out")
			nodes_id_in_path = [node_id for node_id in path]

			flux_from_component, mean_energy = runtime_modules[root_node_id].emission_lines(
				e_low, e_high
			)

			multiplicative_nodes = []

			# Search all multiplicative components connected to this node
			# and apply them at mean energy
			for node_id in nodes_id_in_path[::-1]:
				multiplicative_nodes.extend(
					[node_id for node_id in self.find_multiplicative_components(node_id)]
				)

			for mul_node in multiplicative_nodes:
				flux_from_component *= runtime_modules[mul_node].continuum(mean_energy)

			# if energy_flux:
			# 	fine_structures_flux += trapezoid(
			# 		flux_from_component * energies_to_integrate,
			# 		x=jnp.log(energies_to_integrate),
			# 		axis=0,
			# 	)

			else:
				fine_structures_flux += flux_from_component

		return continuum_flux + fine_structures_flux
	
	@property
	def params(self):
		# return self.transformed_func_photon.init(None, jnp.ones(10), jnp.ones(10))
		return self.transformed_func_photon.init(None, jnp.arange(0, 1024), jnp.arange(1, 1025))




class CustomComponentMetaClass(type(hk.Module)):
	"""
	This metaclass enable the construction of model from components with a simple
	syntax while style enabling the components to be used as haiku modules.
	"""

	def __call__(self, **kwargs) -> CustomSpectralModel:
		"""
		This method enable to use model components as haiku modules when folded in a haiku transform
		function and also to instantiate them as SpectralModel when out of a haiku transform
		"""

		if not base.frame_stack:
			return CustomSpectralModel.from_component(self, **kwargs)

		else:
			return super().__call__(**kwargs)


class CustomModelComponent(hk.Module, ABC, metaclass=CustomComponentMetaClass):
	"""
	Abstract class for model components
	"""

	type: str

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class CustomAdditiveComponent(CustomModelComponent, ABC):
	type = "additive"

	def continuum(self, energy):
		"""
		Method for computing the continuum associated to the model.
		By default, this is set to 0, which means that the model has no continuum.
		This should be overloaded by the user if the model has a continuum.
		"""

		return jnp.zeros_like(energy)

	def emission_lines(self, e_min, e_max) -> (jax.Array, jax.Array):
		"""
		Method for computing the fine structure of an additive model between two energies.
		By default, this is set to 0, which means that the model has no emission lines.
		This should be overloaded by the user if the model has a fine structure.
		"""
		return jnp.zeros_like(e_min), (e_min + e_max) / 2
