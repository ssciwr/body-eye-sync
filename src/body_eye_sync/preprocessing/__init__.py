"""Working out how an experiment's recordings line up, before anything is run.

These stages differ from the pipeline in what they write: the pipeline turns
recordings into results, while preprocessing corrects the experiment's own
definition -- each input's offset, clock scale and lost content -- which the
pipeline then reads. Everything that relates one recording to another depends on
that shared timeline being right, so this comes first.

The functions here take paths and return what they measured; applying it to an
:class:`~body_eye_sync.experiment.experiment.Experiment` is
:mod:`body_eye_sync.experiment.prepare`.
"""
