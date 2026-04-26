grammar mdl;

program
      : (module_def | struct_def | funcdef)* EOF
      ;

struct_def
      : 'struct' IDENT '{' vardecl* '}'
      ;

module_def
      : 'module' IDENT '[' (param (',' param)*)? ']' '{' (vardecl | funcdef)* '}'
      ;

param
      : IDENT ':' type ('=' default=expr)?
      ;

type
      : typename=IDENT ('[' (type_arg (',' type_arg)*)? ']')?   # TypeSpec
      | 'struct' '{' vardecl* '}'                               # AnonType
      ;

type_arg
      : elipsis='...'
      | expr
      ;

constant
      : INTEGER
      | FLOAT
	  | BOOL
      ;

vardecl
      : IDENT ':' type
      ;

funcdef
      : templatespec? type IDENT '(' (param (',' param)*)? ')' '=>' expr ';'
      | templatespec? type IDENT '(' (param (',' param)*)? ')' '{' statement* '}'
      ;

templatespec
      : 'template' '<' vardecl* '>'
      ;

einops
      : einops_spec_full '->' einops_spec_full ('|' (keyvalue (',' keyvalue)*)?)?
      ;

einops_spec_full
      : (einops_atom | '(' einops_atom+ ')')+
      ;

einops_atom
      : IDENT
      | '...'
      | '_'
      | INTEGER
      | '/' INTEGER
      ;

keyvalue
      : IDENT '=' expr
      ;

// Expressions on the top are given more precedent (matched first).
expr
      // Primary Expressions
      : IDENT                                              # VarAccessExpression
      | constant                                           # ConstantExpression
      | '(' expr ')'                                       # ParenExpression
      // Postfix Expressions
      | expr '.' IDENT                                     # AccessExpression
      | expr '[' einops ']'                                # EinopsExpression
      | func=expr '(' (args+=expr (',' args+=expr)*)? ')'  # MethodcallExpression
      // Arithmetic Expressions
      | left=expr bop=('*' | '/' | '@') right=expr         # ArithmeticExpression
      | left=expr bop=('+' | '-') right=expr               # ArithmeticExpression
      // Switch Expression
      | expr '?' (expr '=>' expr)*                         # SwitchExpression
      ;

statement
      : expr ';'                                           # ExpressionStatement
      | (type | 'auto') IDENT '=' expr ';'                 # DeclarationStatement
      | 'return' expr ';'                                  # ReturnStatement
      ;


FLOAT: '-'?[0-9]*'.'[0-9]+ | INTEGER'e'INTEGER;
INTEGER: '-'?[0-9]+;
BOOL: 'true' | 'false';

IDENT: [_a-zA-Z]+;

BLOCKCOMMENT : '/*' .*? '*/' -> channel(HIDDEN);
LINECOMMENT : '//' ~[\r\n]* -> channel(HIDDEN);

WS: [ \r\n\t] -> channel(HIDDEN);