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
              // | funcdef[&$module->graph, $ns] /* create global compute graph */
              //  { }*/
              | EOF
              ;

struct_def returns[Struct value]
            locals[Namespace ns]
            @init{$ns = Namespace::baseNamespace();}
              : STRUCT IDENT LBRACE vardecl[&empty, &$ns]* RBRACE
                { $value.name = $IDENT.text; }
              ;

module_def returns[Module value]
            locals[Namespace ns]
            @init{$ns = Namespace::baseNamespace();}
              : MODULE IDENT LPAREN param_list[&$value.graph, &$ns] RPAREN LBRACE module_entry[&$value, &$ns]* RBRACE
                { $value.name = $IDENT.text; }
              ;

param_list[ComputegraphRef graph, Namespace* ns]
              : ((param[$graph, $ns] COMMA)* param[$graph, $ns])?
                { /**\todo implement**/ }
              ;

param[ComputegraphRef graph, Namespace* ns]
              : IDENT COLON type[$graph, $ns]
                { ns->put($IDENT.text, &$graph->addInputNode($IDENT.text, $type.dtype)); /* \todo For Modules this should not be an input node but an ArgumentNode */ }
              ;

type[ComputegraphRef graph, Namespace* ns]
        returns[DataType dtype]
              : type_specifier[$ns] typearg_specifier[$graph, $ns]
                { $dtype = setTypeArgs($type_specifier.dtype, $typearg_specifier.specs).value(); /* \todo handle error */ }
              | type_specifier[$ns]
                { $dtype = $type_specifier.dtype; }
              ;

type_arg_list[ComputegraphRef graph, Namespace* ns]
    returns[std::vector<TypeArg> typeargs]
              : (args+=type_arg[$graph, $ns] (COMMA args+=type_arg[$graph, $ns])*)?
                {
                  for (const auto& a : $args)
                    $typeargs.emplace_back(a->arg);
                }
              ;

type_arg[ComputegraphRef graph, Namespace* ns]
    returns[TypeArg arg]
              : ELIPSIS
                { $arg = mdl::Elipsis{}; }
              | expr[$graph, $ns]
                {
                  if ($expr.node->operation == Operation::Const) {
                    $arg = $expr.node->constValue();
                  } else if ($expr.node->operation == Operation::Argument) {
                    $arg = $expr.node;
                  } else {
                    /** \todo write descriptive error message **/
                    assert(false);
                  }
                }
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
                { $value = VarDesc{/*.value=*/$IDENT.text, /*.node=*/ns->put($IDENT.text, &$graph->addArgumentNode($IDENT.text, $type.dtype))}; }
              | STRUCT LBRACE vardecl[$graph, $ns]* RBRACE IDENT
                { /**\todo implement**/ }
              ;

funcdef[ComputegraphRef context, Namespace* outer_ns]
        returns[FuncDesc value]
        locals[Namespace ns]
        @init{$value.graph = &context->addSubgraph(); $ns = $outer_ns->createSubnamespace();}
              : IDENT LPAREN param_list[$value.graph, &$ns] RPAREN EQGR expr[$value.graph, &$ns] SEMICOLON
                { $value.name = $IDENT.text; $value.graph->setOutputNode($expr.node); }
              | IDENT LPAREN param_list[$value.graph, &$ns] RPAREN LBRACE statement[$value.graph, &$ns]* RBRACE
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
    returns[const Node* node]
              : IDENT
                { $node = ns->get<const Node*>($IDENT.text); }
              | constant
                { $node = &$graph->addConstNode($constant.value); }
              | LPAREN expr[$graph, $ns] RPAREN
                { $node = $expr.node; }
              ;

expr_postfix[ComputegraphRef graph, Namespace* ns]
    returns[const Node* node]
              : expr_primary[$graph, $ns] (DOT IDENT | LPAREN funcarglist[$graph, $ns] RPAREN | LBRACK einops[$graph, $ns] RBRACK)*
                { $node = $expr_primary.node; /** \todo implement properly **/}
              ;

expr_mult[ComputegraphRef graph, Namespace* ns]
    returns[const Node* node]
              : expr_postfix[$graph, $ns] '*' expr_mult[$graph, $ns]
                { $node = &$graph->addOpNode(Operation::MULT, {$expr_postfix.node, $expr_mult.node}); }
              | expr_postfix[$graph, $ns] SLASH expr_mult[$graph, $ns]
                { $node = &$graph->addOpNode(Operation::DIV, {$expr_postfix.node, $expr_mult.node}); }
              | expr_postfix[$graph, $ns] '@' expr_mult[$graph, $ns]
                { $node = &$graph->addOpNode(Operation::MATMULT, {$expr_postfix.node, $expr_mult.node}); }
              | expr_postfix[$graph, $ns]
                { $node = $expr_postfix.node; }
              ;

expr_add[ComputegraphRef graph, Namespace* ns]
    returns[const Node* node]
              : expr_mult[$graph, $ns] PLUS expr_add[$graph, $ns]
                { $node = &$graph->addOpNode(Operation::ADD, {$expr_mult.node, $expr_add.node}); }
              | expr_mult[$graph, $ns] MINUS expr_add[$graph, $ns]
                { $node = &$graph->addOpNode(Operation::SUB, {$expr_mult.node, $expr_add.node}); }
              | expr_mult[$graph, $ns]
                { $node = $expr_mult.node; }
              ;

expr_switch[ComputegraphRef graph, Namespace* ns]
    returns[const Node* node]
              : expr_add[$graph, $ns]
                { $node = $expr_add.node; }
              | expr_add[$graph, $ns] QMARK switch_alternative[$graph, $ns]*
              ;

expr[ComputegraphRef graph, Namespace* ns]
    returns[const Node* node]
              : expr_switch[$graph, $ns]
                { $node = $expr_switch.node; }
              ;

switch_alternative[ComputegraphRef graph, Namespace* ns]
    returns[const Node* node]
              : IDENT EQGR expr[$graph, $ns]
              ;

type_specifier[Namespace* ns]
    returns[DataType dtype]
              : IDENT
                { $dtype = $ns->getType($IDENT.text); }
              ;

typearg_specifier[ComputegraphRef graph, Namespace* ns]
    returns[std::vector<TypeArg> specs]
              : LBRACK type_arg_list[$graph, $ns] RBRACK
                { $specs = $type_arg_list.typeargs; }
              ;

declaration[ComputegraphRef graph, Namespace* ns]
              : (type[$graph, $ns] | AUTO) IDENT EQ expr[$graph, $ns]
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