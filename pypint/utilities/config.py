# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from os.path import expandvars, expanduser, abspath, dirname, isfile
from sys import stdout, stderr

from configobj import ConfigObj, ConfigObjError

from pypint import __file__ as pypint_path


_CONFIG_SPEC_FILE = abspath(dirname(pypint_path)) + '/utilities/config.spec'
_DEFAULT_CONFIG_FILE = abspath(dirname(pypint_path)) + '/default_config.conf'

_CONFIG_READ = False

_CONFIG = None


def config(config_file=None):
    """Returns configuration object

    By default, *PyPinT* looks in the current user's home directory (i.e. ``$HOME``) for a file called ``.pypint.conf``.

    In case the environment variable ``$PYPINT_CONFIG`` is set, its value is used as the absolute path and file name
    of the config file (cf. ``config_file`` parameter).

    Validation of the given config file is done against the configspec file in PyPinT's installation path in
    ``utilities/config.spec``.

    In case no config file is present, the default values as stated in ``$PYPINT_ROOT/default_config.conf`` file are
    used.

    Parameters
    ----------
    config_file : :py:class:`str`
        *(optional)*
        The absolute path and file name of the config file to use.
        If given, it will override all defaults.

    Returns
    -------
    config : :py:class:`ConfigObj`
        Configuration object parsed from the config file.
    """
    global _CONFIG
    global _CONFIG_READ

    if not _CONFIG_READ:
        print("Configuration has not yet been loaded. Doing it now.", file=stdout)
        _CONFIG = _get_config(config_file=config_file)
        _CONFIG_READ = True

    return _CONFIG


def _get_config(config_file=None):
    """Reads in the configuration from a file

    Parameters
    ----------
    config_file : :py:class:`str`
        see :py:func:`.get_config`

    Returns
    -------
    config : :py:class:`ConfigObj`
        Configuration object parsed from the config file.

    Raises
    ------
    IOError
        if the config file could not be found
    ConfigObjError
        on parsing errors
    """
    _config_file = expandvars('$PYPINT_CONFIG') if config_file is None else config_file
    if _config_file == '$PYPINT_CONFIG':
        # default config file location has not been overridden by user
        # -> taking default
        _config_file = expanduser('~') + '/.pypint.conf'

    if not isfile(_config_file):
        print("WARNING: Configuration file '%s' not found. Using defaults." % _config_file, file=stdout)
        print("WARNING: Configuration file '%s' not found. Using defaults." % _config_file,file=stderr)
        _config_file = _DEFAULT_CONFIG_FILE

    _config_spec = ConfigObj(_CONFIG_SPEC_FILE, encoding='UTF-8', interpolation=False, list_values=False,
                             _inspec=True)

    try:
        return ConfigObj(_config_file, interpolation=False, file_error=True, raise_errors=True, encoding='UTF-8',
                         configspec=_config_spec)
    except IOError as err:
        # Shouldn't happend!
        print("Configuration file '%s' not found." % _config_file, file=stdout)
        print("Configuration file '%s' not found. (Shouldn't reached here, thus something really bad happend!)"
              % _config_file, file=stderr)
        raise err
    except ConfigObjError as err:
        print("Error occured while parsing the config file at '%s'. Check syntax." % _config_file, file=stdout)
        print("Error occured while parsing the config file at '%s'. Check syntax." % _config_file, file=stderr)
        raise err


__all__ = [
    'config'
]
