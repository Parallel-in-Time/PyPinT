# Configuration Specification

# Options for the logging behaviour
[Logger]
    [[Stderr]]
    enable = boolean(default=True)
    level = option('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', default='WARNING')
    format_string = string(default='[{record.level_name: <8s}] {record.module:s}.{record.func_name:s}(): {record.message:s}')
    bubble = boolean(default=True)

    [[Stdout]]
    enable = boolean(default=True)
    level = option('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', default='INFO')
    format_string = string(default='[{record.level_name: <8s}] {record.message:s}')
    bubble = boolean(default=True)

    [[File]]
    enable = boolean(default=False)
    file_name_format = string(default='{:%Y-%m-%d_%H-%M-%S}_debug.log')
    level = option('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', default='DEBUG')
    format_string = string(default='[{record.time}] [{record.level_name: <8s}] <{record.process}.{record.thread}> {record.module:s}.{record.func_name:s}():{record.lineno:d}: {record.message:s}')

    [[numpy]]
    precision = integer(default=4)
    linewidth = integer(default=80)
