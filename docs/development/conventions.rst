Conventions
-----------

Code Style
^^^^^^^^^^

Clean, highly-readable code is important, especially for open-source projects
with several developers and contributors.

But you and I might have different opinions about the best way to format our
code. You might like separating function arguments on new lines, while I might
like keeping them all on the same line. Or you might like two lines between
class definitions while I prefer one.

It's a waste of our time to fight about code style - or even to think about it
at all during development - so we give up style control and use an
autoformatter.

`Black <https://github.com/psf/black>`_ is an opinionated autoformatter for
Python code with few configuration options. We use it on all code, every
commit.

We also use `isort <https://pycqa.github.io/isort/>`_ for automatic import
sorting.

Linting
^^^^^^^

Code linting helps catch errors, clean up dead code, and anticipate edge cases
early. 

`Flake8 <https://flake8.pycqa.org/en/latest/>`_ is a popular python linter.
We use it on all code, every commit.

Pre-Commit Hooks
^^^^^^^^^^^^^^^^

    Wow, as a developer that's a lot of stuff I have to do before I push any
    code... What a pain.

Luckily, we have `Pre-Commit <https://pre-commit.com/>`_ hooks for code style,
linting, and type checking that you can install in your development
environment (see :ref:`development-setup`).

We also have CI set up to run these on every commit and send you angry emails
if they don't pass - so you should probably use the Pre-Commit hooks. Pull
requests not passing CI checks will be ignored until they do.

Code Comments
^^^^^^^^^^^^^

We don't require documents on all code, all of the time. If you're going to add
comments to your code, it's nice to do them in a machine-readable style so we
can generate documentation from them later.

We use `Google style python docstrings
<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
for commenting/documenting code

It's a very human and machine readable comment style that sphinx autodoc
supports out of the box.

Feature Branching
^^^^^^^^^^^^^^^^^

    Okay cool, now I can write a bunch of code and push it directly to the
    ``main`` branch, right?

No - don't push directly to ``main``. We don't want to see all your gross, work
in progress code.

Create your own branch, do whatever you want there - break style conventions,
create a disgusting hairball of commit history and merges, push half-baked
ideas, whatever you want.

Just don't push directly to ``main``.

    What if I need to fix something urgently?

What if you break something else by trying to fix it? Then you have to push
another urgent fix and we have two commits when we should only have one. One of
those commits also probably contains the word "oops".

It's really not that hard to put it in a branch and open a pull request.

Linear History & Squash Merging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Okay my feature is complete and I'm ready to put it in ``main``.

Great! Open a pull request, if you want review or think anyone will want to
review it, get review.

You probably did a lot of work on your feature branch, maybe in a bunch of
different commits as you made progress. You might have even gone back an forth
on the design a bit, or fixed some things during review.

No one needs to see all those commits on ``main``. It clutters the history. And
we especially don't want to see a merge commit - gross.

Squash merge - that way we keep a clean linear history on main that is easy to
follow. There's a nice easy button for this on GitHub so you don't have to know
how to do it in git if you don't want.

One single feature or fix from a branch means one commit to main no matter how
many commits on your feature branch it took you to get right.

Conventional Commits
^^^^^^^^^^^^^^^^^^^^

    Okay, I'll squash my feature when I merge. What should I make the commit
    message?

There's actually a standard for this too that we follow - it's called
`Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`_. It
looks like this::

    <type>: <description>

Where ``<type>`` can be any of ``feat``, ``fix``, ``build``, ``chore``, ``ci``,
``docs``, ``style``, ``refactor``, ``perf``, or ``test``.

Here's an example for your feature branch::

    feat: add a new widget

.. note:: We currently don't use feature scopes or breaking change markers.
