class RePattern(object):
    TIMEPATTERN = '\[\d{4}-\d{2}-\d{2}\s+([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]]'
    SPEAKERPATTERN = '[^\[]\<{1}([a-zA-Z0-9\-\~\_\.]+)>[^]]'
    COLONReplyPATTREN = '(([a-zA-Z0-9])+):'
    ATREPLYPATTERN = '@(([a-zA-Z0-9])+)'
    URLPATTERN = '(?P<url>https?://[^\s]+)'
    NOFORMATREGION = '(\{noformat\})([\s\S]*?)?(\{noformat\})'
    CODEREGION = '(\{code(.*?)\})([\s\S]*?)?(\{code(.*?)\})'
    COLOREDREGION = '(\{color(.*?)\})([\s\S]*?)?(\{color(.*?)\})'
    NOFORMATTAG = '\{noformat\}'
    CODETAG = '\{code(.*?)\}'
    COLOREDTAG = '\{color(.*?)\}'
    TOKENIZPATTERN = r'''(?x)    # set flag to allow verbose regexps
            (?:[a-zA-Z]\.)+       # abbreviations, e.g. U.S.A.   
           |\$?\d+(?:\.\d+)*%?  # currency and percentages, e.g. $12.40, 82%
           |\w+(?:[-'.]\w+)*   # words with optional internal hyphens\
           |\.\.\.            # ellipsis
           |(?:[.,;"?():-_`])  # these are separate tokens; includes ], [
         '''


class DialogRef(object):
    REF = ['[<-LINK->]', '[<-CODE->]', '[<-ISSUE->]']


class CodePattern(object):
    JAVACODEPATTERN = {'import': r'(?m)^\s*import.*;$',
                       'package': r'(?m)^package.*;$',
                       'singlecomment': r'(?m)\s*\/\/.*?[\n\r]',
                       'multicomment': r'(?m)(?s)\s*(\/\*).*?(\*\/)',
                       'class': r'(?m)^.*?class.*?([\n\r])?\{',
                       'assignment': r'(?m)^.*=.*;$',
                       'ifstatement': r'(?m)^.*?if\s*\(.*?\)\s*\{',
                       'elsestatement': r'(?m)^.*?else\s*\{',
                       'functiondef': r'(?m)^.*?([a-zA-Z_][a-zA-Z0-9_])+\s*\(.*?\)\s*?\{',
                       'functionall': r'(?m)^.*\(.*?\).*?;$'}
    JAVACODEPATTERNOPTIONS = {'import': '',
                              'package': '',
                              'singlecomment': '',
                              'multicomment': '',
                              'class': 'MATCH',
                              'assignment': '',
                              'ifstatement': 'MATCH',
                              'elsestatement': 'MATCH',
                              'functiondef': 'MATCH',
                              'functionall': ''}


class StackTracePattern(object):
    JAVAEXCEPTION = r'(([\w<>\$_]+\.)+[\w<>\$_]+((.\s*)Error|(.\s*)Exception){1}(\s|:))'
    JAVAREASON = r'((:?([\s\S]*?)?)(at\s+([\w<>\$_]+\.)+[\w<>\$_]+\s*\(.+?\.java(:)?(\d+)?\)))'
    JAVATRACE = r'(\s*?at\s+([\w<>\$_\s]+\.)+[\w<>\$_\s]+\s*(.+?\.java(:)?(\d+)?\))*)'
    JAVACAUSE = r'(Caused by:).*?(Exception|Error)(.*?)(\s+at.*?\(.*?:\d+\))+'
    JAVASTACKTRACE = JAVAEXCEPTION + JAVAREASON + '?' + JAVATRACE + '*'
    # JAVASTACKTRACE=(([\w<>\$_]+\.)+[\w<>\$_]+((.\s*)Error|(.\s*)Exception){1}(\s|:))((:?([\s\S]*?)?)(at\s+([\w<>\$_]+\.)+[\w<>\$_]+\s*\(.+?\.java(:)?(\d+)?\)))?(\s*?at\s+([\w<>\$_\s]+\.)+[\w<>\$_\s]+\s*(.+?\.java(:)?(\d+)?\))*)*
