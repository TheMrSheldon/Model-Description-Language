grammar mdl;

program
      : (import_stmt | module_def | struct_def | funcdef)* EOF
      ;

// `import a.b;` imports the definitions of `a/b.dlm` or, if `a/b` is a directory, of all files in it.
import_stmt
      : 'import' path+=IDENT ('.' path+=IDENT)* ';'
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
      // `masked` marks a masked tensor (dense values plus a mask of the valid elements, see the masking pass)
      : masked='masked'? typename=IDENT ('[' (type_arg (',' type_arg)*)? ']')?   # TypeSpec
      | 'struct' '{' vardecl* '}'                               # AnonType
      // Array of `element` for var = lo, ..., hi-1 (e.g. a stack of modules)
      | '[' element=type 'for' var=(IDENT | '_') 'in' lo=expr '..' hi=expr ']' # ComprehensionType
      // A compile-time array of `size` values (e.g. a hyperparameter `ch: array[uint; 5]`); not a tensor
      | 'array' '[' element=type ';' size=expr ']'          # ArrayType
      ;

type_arg
      : elipsis='...'
      // A named hyperparameter (designated-initializer style): `Linear[768, 3, .bias=false]`
      | '.' name=IDENT '=' value=expr
      | expr
      ;

constant
      : INTEGER
      | FLOAT
	  | BOOL
      ;

vardecl
      : annotation* IDENT ':' type
      ;

// `@name` or `@name(args...)`, e.g. `@conditional(bias)` before a member
annotation
      : '@' name=IDENT ('(' (args+=expr (',' args+=expr)*)? ')')?
      ;

funcdef
      : templatespec? rettype IDENT '(' (param (',' param)*)? ')' '=>' expr ';'
      | templatespec? rettype IDENT '(' (param (',' param)*)? ')' '{' statement* '}'
      ;

// A function returns one value or (in parentheses) multiple values.
rettype
      : type
      | '(' type (',' type)* ')'
      ;

templatespec
      : 'template' '<' vardecl* '>'
      ;

// An index of an array (one integer) or of a tensor (NumPy-like: integers select and remove an axis, `:` keeps an axis,
// `...` stands for the axes that are not listed).
index_item
      : '...'                                              # IndexEllipsis
      | ':'                                                # IndexAll
      | expr                                               # IndexValue
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
      // `_` discards a value (`auto a, _ = f(x);`, `for (_ in ...)`); using it is an error
      | '_'                                                # DiscardExpression
      | constant                                           # ConstantExpression
      | '(' expr ')'                                       # ParenExpression
      // A compile-time array (e.g. a hyperparameter value `[64, 128, 256]`)
      | '[' (elements+=expr (',' elements+=expr)*)? ']'    # ArrayLiteralExpression
      // Postfix Expressions
      | expr '.' IDENT                                     # AccessExpression
      | expr '[' einops ']'                                # EinopsExpression
      | base=expr '[' indices+=index_item (',' indices+=index_item)* ']' # IndexExpression
      | func=expr '(' (args+=expr (',' args+=expr)*)? ')'  # MethodcallExpression
      // Prefix Expressions
      | '-' expr                                           # NegateExpression
      // Arithmetic Expressions
      | left=expr bop=('*' | '/' | '@') right=expr         # ArithmeticExpression
      | left=expr bop=('+' | '-') right=expr               # ArithmeticExpression
      // Compile-time conditional: only the chosen branch is elaborated
      | <assoc=right> cond=expr '?' then=expr ':' otherwise=expr # ConditionalExpression
      ;

statement
      : expr ';'                                           # ExpressionStatement
      // Without initializer (only with a tensor type `T[n, ...]`): a tensor whose n slots `name[i]` are assigned one by
      // one (e.g. per layer in a loop); every slot must be assigned before the tensor (or the slot) is used.
      | (type | 'auto') names+=(IDENT | '_') (',' names+=(IDENT | '_'))* ('=' expr)? ';' # DeclarationStatement
      | 'return' values+=expr (',' values+=expr)* ';'       # ReturnStatement
      // Rebinds a variable of an enclosing scope (e.g. the state carried through a loop)
      | name=IDENT '=' expr ';'                             # AssignStatement
      // Assigns a slot of a tensor declared without initializer (a compile-time index)
      | name=IDENT '[' index=expr ']' '=' value=expr ';'      # SlotAssignStatement
      // Compile-time loop: the body is expanded for var = lo, ..., hi-1
      | 'for' '(' var=(IDENT | '_') 'in' lo=expr '..' hi=expr ')' '{' statement* '}' # ForStatement
      ;


// Note: literals are unsigned; negative numbers are parsed as NegateExpression (so that `x-1` is a subtraction).
FLOAT: [0-9]* '.' [0-9]+ EXPONENT? | [0-9]+ EXPONENT;
fragment EXPONENT: [eE] [+-]? [0-9]+;
INTEGER: [0-9]+;
BOOL: 'true' | 'false';

IDENT: [_a-zA-Z][_a-zA-Z0-9]*;

BLOCKCOMMENT : '/*' .*? '*/' -> channel(HIDDEN);
LINECOMMENT : '//' ~[\r\n]* -> channel(HIDDEN);

WS: [ \r\n\t] -> channel(HIDDEN);