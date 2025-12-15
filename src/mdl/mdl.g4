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
            locals[Namespace ns]
              : STRUCT IDENT LBRACE vardecl[&empty, &$ns]* RBRACE
                { $value.name = $IDENT.text; }
              ;

module_def returns[Module value]
            locals[Namespace ns]
              : MODULE IDENT LPAREN param_list[&$value.graph, &$ns] RPAREN LBRACE module_entry[&$value, &$ns]* RBRACE
                { $value.name = $IDENT.text; }
              ;

param_list[ComputegraphRef graph, Namespace* ns]
              : ((param[$graph, $ns] COMMA)* param[$graph, $ns])?
              ;

param[ComputegraphRef graph, Namespace* ns]
              : IDENT COLON type[$graph, $ns]
                { ns->put($IDENT.text, $graph->addInputNode()); }
              ;

type[ComputegraphRef graph, Namespace* ns]
              : type_specifier typearg_specifier[$graph, $ns]?
              ;

type_arg_list[ComputegraphRef graph, Namespace* ns]
              : (args+=type_arg[$graph, $ns] (COMMA args+=type_arg[$graph, $ns])*)?
              ;

type_arg[ComputegraphRef graph, Namespace* ns]
              : ELIPSIS
              | expr[$graph, $ns]
              ;

constant returns[std::variant<double, int> value]
              : INTEGER
                {$value = $INTEGER.int;}
              | FLOAT
                {$value = std::stod($FLOAT.text);}
              ;

module_entry[Module* module, Namespace* ns]
              : vardecl[&$module->graph, $ns]
                { $module->variables.emplace_back($vardecl.value); }
              | funcdef[&$module->graph, $ns]
                { $module->functions.emplace_back($funcdef.value); }
              ;

vardecl[ComputegraphRef graph, Namespace* ns]
        returns[VarDesc value]
              : IDENT COLON type[$graph, $ns]
                { $value = VarDesc{/*.value=*/$IDENT.text, /*.node=*/&$graph->addArgumentNode()}; }
              | STRUCT LBRACE vardecl[$graph, $ns]* RBRACE IDENT
              ;

funcdef[ComputegraphRef context, Namespace* outer_ns]
        returns[FuncDesc value]
        locals[Namespace ns]
        @init{$value.graph = &context->addSubgraph(); $ns = {*$outer_ns};}
              : IDENT LPAREN param_list[$context, &$ns] RPAREN EQGR expr[$value.graph, &$ns]
                { $value.name = $IDENT.text; }
              | IDENT LPAREN param_list[$context, &$ns] RPAREN LBRACE statement[$value.graph, &$ns]* RBRACE
                { $value.name = $IDENT.text; }
              ;

funcarglist[ComputegraphRef graph, Namespace* ns]
              : (expr[$graph, $ns] (COMMA expr[$graph, $ns] COMMA)*)?
              ;

einops[ComputegraphRef graph, Namespace* ns]
              : einops_spec_full RARROW einops_spec_full (PIPE keyvaluelist[$graph, $ns])?
              ;

einops_spec_full : (einops_spec | LPAREN einops_spec+ RPAREN)+
                 ;

einops_spec   : IDENT
              | ELIPSIS
              | UNDERSCORE
              | INTEGER
              | SLASH INTEGER
              ;

keyvaluelist[ComputegraphRef graph, Namespace* ns]
              : ((keyvalue[$graph, $ns] ',')* keyvalue[$graph, $ns])?
              ;

keyvalue[ComputegraphRef graph, Namespace* ns]
              : IDENT EQ expr[$graph, $ns]
              ;

expr_primary[ComputegraphRef graph, Namespace* ns]
              : IDENT
              | constant
              | LPAREN expr[$graph, $ns] RPAREN
              ;

expr_postfix[ComputegraphRef graph, Namespace* ns]
              : expr_primary[$graph, $ns] (DOT IDENT | LPAREN funcarglist[$graph, $ns] RPAREN | LBRACK einops[$graph, $ns] RBRACK)*
              ;

expr_mult[ComputegraphRef graph, Namespace* ns]
              : expr_postfix[$graph, $ns] ('*' expr_postfix[$graph, $ns] | SLASH expr_postfix[$graph, $ns] | '@' expr_postfix[$graph, $ns])*
              ;

expr_add[ComputegraphRef graph, Namespace* ns]
              : expr_mult[$graph, $ns] (PLUS expr_mult[$graph, $ns] | MINUS expr_mult[$graph, $ns])*
              ;

expr_switch[ComputegraphRef graph, Namespace* ns]
              : expr_add[$graph, $ns]
              | expr_add[$graph, $ns] QMARK switch_alternative[$graph, $ns]*
              ;

expr[ComputegraphRef graph, Namespace* ns]
              : expr_switch[$graph, $ns]
              ;

switch_alternative[ComputegraphRef graph, Namespace* ns]
              : IDENT EQGR expr[$graph, $ns]
              ;

type_specifier : IDENT
               ;

typearg_specifier[ComputegraphRef graph, Namespace* ns]
                  : LBRACK type_arg_list[$graph, $ns] RBRACK
                  ;

declaration[ComputegraphRef graph, Namespace* ns]
                  : (type_specifier typearg_specifier[$graph, $ns] | AUTO) IDENT EQ expr[$graph, $ns]
                  ;

statement[ComputegraphRef graph, Namespace* ns]
              : expr[$graph, $ns] SEMICOLON
              | declaration[$graph, $ns] SEMICOLON
              | RETURN expr[$graph, $ns] SEMICOLON
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