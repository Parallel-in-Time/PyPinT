Release 0.0.2 (04/03/2014)
--------------------------

major rework without any additional fancy features but cleaner API

- Core

  * thorough redesign

- Solvers

  * refactor SDC solver
  * debug and fix Implicit and Semi-Implicit SDC solver cores

- Documentation

  * improve API documentation

- Examples

  * adjust examples to changes in core

- Plugins

  * adjust Analyser and Plotter plugins

- Internal Structure

  * move computation of reductions into ThresholdChecker
  * move solver's state into object hierarchy to pass around
  * save reductions of error and solution to IterativeSolution objects
  * update / write / extend unit tests


Release 0.0.1 (2013-11-26)
--------------------------

* thorough framework design for high flexibility and easy understanding
* basic implementation of explicit SDC
* basic solution analyzer and solution plotter
* simple example for applying SDC
* unit tests for core functionality
