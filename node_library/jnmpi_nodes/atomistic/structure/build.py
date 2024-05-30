from pyiron_atomistics import _StructureFactory
from pyiron_workflow.workflow import Workflow


Bulk = Workflow.wrap.as_function_node("structure")(_StructureFactory().bulk)
Bulk.__name__ = "Bulk"
Bulk.__module__ = __name__


@Workflow.wrap.as_macro_node("structure")
def CubicBulkCell(
    wf, element: str, cell_size: int = 1, vacancy_index: int | None = None
):
    from node_library.jnmpi_nodes.atomistic.structure.transform import (
        CreateVacancy,
        Repeat,
    )

    wf.bulk = Bulk(name=element, cubic=True)
    wf.cell = Repeat(structure=wf.bulk, repeat_scalar=cell_size)

    wf.structure = CreateVacancy(structure=wf.cell, index=vacancy_index)
    return wf.structure


nodes = [
    Bulk,
    CubicBulkCell,
]
