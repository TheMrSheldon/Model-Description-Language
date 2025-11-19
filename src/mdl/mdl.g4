grammar mdl;

@parser::header {
    #include <mdl/parser/definitions.hpp>
}
@parser::source {
  Computegraph empty;
}

program returns[Program value]
              : program_body[&$value]
              ;

program_body[Program* prog]
              : module_def program_body[$prog]
                { $prog->definitions.try_emplace($module_def.value.name, std::move($module_def.value)); }
              | struct_def program_body[$prog]
                { $prog->definitions.try_emplace($struct_def.value.name, std::move($struct_def.value)); }
              | EOF
              ;

struct_def returns[Struct value]
              : STRUCT IDENT LBRACE vardecl[&empty]* RBRACE
              ;

module_def returns[Module value]
              : MODULE IDENT LPAREN param_list[&$value.graph] RPAREN LBRACE module_entry[&$value]* RBRACE
              ;

param_list[ComputegraphRef graph]
              : ((param[$graph] COMMA)* param[$graph])?
              ;

param[ComputegraphRef graph]
              : IDENT COLON type[$graph]
              ;

type[ComputegraphRef graph]
              : type_specifier typearg_specifier[$graph]?
              ;

type_arg_list[ComputegraphRef graph]
              : (args+=type_arg[$graph] (COMMA args+=type_arg[$graph])*)?
              ;

type_arg[ComputegraphRef graph]
              : ELIPSIS
              | expr[$graph]
              ;

constant returns[std::variant<double, int> value]
              : INTEGER
                {$value = $INTEGER.int;}
              | FLOAT
                {$value = std::stod($FLOAT.text);}
              ;

module_entry[Module* module]
              : vardecl[&$module->graph]
                { $module->variables.emplace_back($vardecl.value); }
              | funcdef[&$module->graph]
                { $module->functions.emplace_back($funcdef.value); }
              ;

vardecl[ComputegraphRef graph]
        returns[VarDesc value]
              : IDENT COLON type[$graph]
                { $value = VarDesc{/*.value=*/$IDENT.text, /*.node=*/&$graph->addArgumentNode()}; }
              | STRUCT LBRACE vardecl[$graph]* RBRACE IDENT
              ;

funcdef[ComputegraphRef context]
        returns[FuncDesc value]
        @init{$value.graph = &context->addSubgraph();}
              : IDENT LPAREN param_list[$context] RPAREN EQGR expr[$value.graph]
                { $value.name = $IDENT.text; }
              | IDENT LPAREN param_list[$context] RPAREN LBRACE statement[$value.graph]* RBRACE
                { $value.name = $IDENT.text; }
              ;

funcarglist[ComputegraphRef graph]
              : (expr[$graph] (COMMA expr[$graph] COMMA)*)?
              ;

einops[ComputegraphRef graph]
              : einops_spec_full RARROW einops_spec_full (PIPE keyvaluelist[$graph])?
              ;

einops_spec_full : (einops_spec | LPAREN einops_spec+ RPAREN)+
                 ;

einops_spec   : IDENT
              | ELIPSIS
              | UNDERSCORE
              | INTEGER
              | SLASH INTEGER
              ;

keyvaluelist[ComputegraphRef graph]
              : ((keyvalue[$graph] ',')* keyvalue[$graph])?
              ;

keyvalue[ComputegraphRef graph]
              : IDENT EQ expr[$graph]
              ;

expr_primary[ComputegraphRef graph]
              : IDENT
              | constant
              | LPAREN expr[$graph] RPAREN
              ;

expr_postfix[ComputegraphRef graph]
              : expr_primary[$graph] (DOT IDENT | LPAREN funcarglist[$graph] RPAREN | LBRACK einops[$graph] RBRACK)*
              ;

expr_mult[ComputegraphRef graph]
              : expr_postfix[$graph] ('*' expr_postfix[$graph] | SLASH expr_postfix[$graph] | '@' expr_postfix[$graph])*
              ;

expr_add[ComputegraphRef graph]
              : expr_mult[$graph] (PLUS expr_mult[$graph] | MINUS expr_mult[$graph])*
              ;

expr_switch[ComputegraphRef graph]
              : expr_add[$graph]
              | expr_add[$graph] QMARK switch_alternative[$graph]*
              ;

expr[ComputegraphRef graph]
              : expr_switch[$graph]
              ;

switch_alternative[ComputegraphRef graph]
              : IDENT EQGR expr[$graph]
              ;

type_specifier : IDENT
               ;

typearg_specifier[ComputegraphRef graph]
                  : LBRACK type_arg_list[$graph] RBRACK
                  ;

declaration[ComputegraphRef graph]
                  : (type_specifier typearg_specifier[$graph] | AUTO) IDENT EQ expr[$graph]
                  ;

statement[ComputegraphRef graph]
              : expr[$graph] SEMICOLON
              | declaration[$graph] SEMICOLON
              | RETURN expr[$graph] SEMICOLON
              ; /* todo: let return set the output node */

RETURN: 'return';
STRUCT: 'struct';
MODULE: 'module';
AUTO: 'auto';

FLOAT: '-'?[0-9]*'.'[0-9]+;
INTEGER: '-'?[0-9]+;

IDENT: [a-zA-Z]+;

BLOCKCOMMENT : '/*' .*? '*/' -> channel(HIDDEN);
LINECOMMENT : '//' ~[\r\n]* -> channel(HIDDEN);

DOT: '.';
EQGR: '=>';
RARROW: '->';
MINUS: '-';
PLUS: '+';
SLASH: '/';
EQ: '=';
PIPE: '|';
QMARK: '?';
SEMICOLON: ';';
COLON: ':';
COMMA: ',';
LPAREN: '(';
RPAREN: ')';
LBRACK: '[';
RBRACK: ']';
LBRACE: '{';
RBRACE: '}';
ELIPSIS: '...';
UNDERSCORE: '_';

WS: [ \r\n\t] -> channel(HIDDEN);