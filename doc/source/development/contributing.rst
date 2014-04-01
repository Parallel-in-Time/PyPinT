Contributing
============

Branching and Tagging Model
---------------------------

We use the `git-flow`_ workflow to that extend, that branch names(paces) have a certain meaing:

``master``
    tip of the ``master`` branch is always the latest stable release

``development``
    tip of the ``development`` branch is the current state of development and not expected to be stable or even usable

``feature/*``
    various feature branches are used to implement new features and should be based off the ``development`` branch

``release/*``
    a release branch is created from the ``development`` branch and used to prepare a new release and will be merged
    into ``master``

``hotfix/*``
    hotfix branches are based off ``master`` or ``development`` to fix important and severe bugs and should be merged
    into ``development`` and ``master`` as soon as possible

Releases and release candidates are tagged in the form ``release-X.Y.Z(-RCa)``, where ``X``, ``Y``, and ``Z`` specify
the version with respect to `semantic versioning`_ and ``a`` the number of the release candidate of that version.

.. _git-flow: http://nvie.com/posts/a-successful-git-branching-model/
.. _semantic versioning: http://semver.org/


Commit Messages
---------------

To ease browsing the proejct's history, we try to keep our commit messages clean and descriptive.
Please try to follow the following rules as best as possible:

* Commit Title must not be longer than 50 characters

  If applicable, the title should start with a category name (such as '``docu``', '``tests``', '``solvers``', ...)
  followed by a colon (e.g. "*docu: add usage examples for SDC solver*").

* Commit Description must have line wraps at 72 characters

* Please *sign* your commits (i.e. use ``git commit -s``)

  This automatically appends a line of the form "*Signed-off-by: Torbjörn Klatt <t.klatt@fz-juelich.de>*" to the end
  of the commit message.



How to Implement a New Feature?
-------------------------------

1. create a fork/clone

2. switch to the ``development`` branch and pull in the latest changes

3. create a new branch ``feature-XYZ`` where ``XYZ`` is a short title of your planned feature
   (word seperation should be done with underscores, e.g. ``feature/my_awesome_feature``)

4. hack and write Unit Tests

5. commit

6. repeat steps 4 and 5 until you feel your feature is in an almost usable state and most of the unit tests pass

7. write documentation for your feature

8. push your feature branch

9. stay tuned on reviews, remarks and suggestions by the other developers
