// Generated from /workspaces/model-description-language/src/mdl/mdl.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue"})
public class mdlParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.13.1", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, T__6=7, T__7=8, T__8=9, 
		T__9=10, T__10=11, T__11=12, T__12=13, T__13=14, T__14=15, T__15=16, T__16=17, 
		T__17=18, T__18=19, T__19=20, T__20=21, T__21=22, T__22=23, T__23=24, 
		T__24=25, T__25=26, T__26=27, T__27=28, T__28=29, FLOAT=30, INTEGER=31, 
		IDENT=32, BLOCKCOMMENT=33, LINECOMMENT=34, WS=35;
	public static final int
		RULE_program = 0, RULE_struct_def = 1, RULE_module_def = 2, RULE_param = 3, 
		RULE_type = 4, RULE_type_arg = 5, RULE_constant = 6, RULE_vardecl = 7, 
		RULE_funcdef = 8, RULE_templatespec = 9, RULE_einops = 10, RULE_einops_spec_full = 11, 
		RULE_einops_atom = 12, RULE_keyvalue = 13, RULE_expr = 14, RULE_statement = 15;
	private static String[] makeRuleNames() {
		return new String[] {
			"program", "struct_def", "module_def", "param", "type", "type_arg", "constant", 
			"vardecl", "funcdef", "templatespec", "einops", "einops_spec_full", "einops_atom", 
			"keyvalue", "expr", "statement"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'struct'", "'{'", "'}'", "'module'", "'['", "','", "']'", "':'", 
			"'='", "'...'", "'('", "')'", "'=>'", "';'", "'template'", "'<'", "'>'", 
			"'->'", "'|'", "'_'", "'/'", "'.'", "'*'", "'@'", "'+'", "'-'", "'?'", 
			"'auto'", "'return'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, "FLOAT", "INTEGER", "IDENT", "BLOCKCOMMENT", 
			"LINECOMMENT", "WS"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "mdl.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public mdlParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ProgramContext extends ParserRuleContext {
		public TerminalNode EOF() { return getToken(mdlParser.EOF, 0); }
		public List<Module_defContext> module_def() {
			return getRuleContexts(Module_defContext.class);
		}
		public Module_defContext module_def(int i) {
			return getRuleContext(Module_defContext.class,i);
		}
		public List<Struct_defContext> struct_def() {
			return getRuleContexts(Struct_defContext.class);
		}
		public Struct_defContext struct_def(int i) {
			return getRuleContext(Struct_defContext.class,i);
		}
		public List<FuncdefContext> funcdef() {
			return getRuleContexts(FuncdefContext.class);
		}
		public FuncdefContext funcdef(int i) {
			return getRuleContext(FuncdefContext.class,i);
		}
		public ProgramContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_program; }
	}

	public final ProgramContext program() throws RecognitionException {
		ProgramContext _localctx = new ProgramContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_program);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(37);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 4295000082L) != 0)) {
				{
				setState(35);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,0,_ctx) ) {
				case 1:
					{
					setState(32);
					module_def();
					}
					break;
				case 2:
					{
					setState(33);
					struct_def();
					}
					break;
				case 3:
					{
					setState(34);
					funcdef();
					}
					break;
				}
				}
				setState(39);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(40);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Struct_defContext extends ParserRuleContext {
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public List<VardeclContext> vardecl() {
			return getRuleContexts(VardeclContext.class);
		}
		public VardeclContext vardecl(int i) {
			return getRuleContext(VardeclContext.class,i);
		}
		public Struct_defContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_struct_def; }
	}

	public final Struct_defContext struct_def() throws RecognitionException {
		Struct_defContext _localctx = new Struct_defContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_struct_def);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(42);
			match(T__0);
			setState(43);
			match(IDENT);
			setState(44);
			match(T__1);
			setState(48);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==IDENT) {
				{
				{
				setState(45);
				vardecl();
				}
				}
				setState(50);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(51);
			match(T__2);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Module_defContext extends ParserRuleContext {
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public List<ParamContext> param() {
			return getRuleContexts(ParamContext.class);
		}
		public ParamContext param(int i) {
			return getRuleContext(ParamContext.class,i);
		}
		public List<VardeclContext> vardecl() {
			return getRuleContexts(VardeclContext.class);
		}
		public VardeclContext vardecl(int i) {
			return getRuleContext(VardeclContext.class,i);
		}
		public List<FuncdefContext> funcdef() {
			return getRuleContexts(FuncdefContext.class);
		}
		public FuncdefContext funcdef(int i) {
			return getRuleContext(FuncdefContext.class,i);
		}
		public Module_defContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_module_def; }
	}

	public final Module_defContext module_def() throws RecognitionException {
		Module_defContext _localctx = new Module_defContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_module_def);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(53);
			match(T__3);
			setState(54);
			match(IDENT);
			setState(55);
			match(T__4);
			setState(64);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==IDENT) {
				{
				setState(56);
				param();
				setState(61);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__5) {
					{
					{
					setState(57);
					match(T__5);
					setState(58);
					param();
					}
					}
					setState(63);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(66);
			match(T__6);
			setState(67);
			match(T__1);
			setState(72);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 4295000066L) != 0)) {
				{
				setState(70);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,5,_ctx) ) {
				case 1:
					{
					setState(68);
					vardecl();
					}
					break;
				case 2:
					{
					setState(69);
					funcdef();
					}
					break;
				}
				}
				setState(74);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(75);
			match(T__2);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ParamContext extends ParserRuleContext {
		public ExprContext default_;
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ParamContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_param; }
	}

	public final ParamContext param() throws RecognitionException {
		ParamContext _localctx = new ParamContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_param);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(77);
			match(IDENT);
			setState(78);
			match(T__7);
			setState(79);
			type();
			setState(82);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__8) {
				{
				setState(80);
				match(T__8);
				setState(81);
				((ParamContext)_localctx).default_ = expr(0);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class TypeContext extends ParserRuleContext {
		public TypeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_type; }
	 
		public TypeContext() { }
		public void copyFrom(TypeContext ctx) {
			super.copyFrom(ctx);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class TypeSpecContext extends TypeContext {
		public Token typename;
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public List<Type_argContext> type_arg() {
			return getRuleContexts(Type_argContext.class);
		}
		public Type_argContext type_arg(int i) {
			return getRuleContext(Type_argContext.class,i);
		}
		public TypeSpecContext(TypeContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class AnonTypeContext extends TypeContext {
		public List<VardeclContext> vardecl() {
			return getRuleContexts(VardeclContext.class);
		}
		public VardeclContext vardecl(int i) {
			return getRuleContext(VardeclContext.class,i);
		}
		public AnonTypeContext(TypeContext ctx) { copyFrom(ctx); }
	}

	public final TypeContext type() throws RecognitionException {
		TypeContext _localctx = new TypeContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_type);
		int _la;
		try {
			setState(108);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case IDENT:
				_localctx = new TypeSpecContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(84);
				((TypeSpecContext)_localctx).typename = match(IDENT);
				setState(97);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__4) {
					{
					setState(85);
					match(T__4);
					setState(94);
					_errHandler.sync(this);
					_la = _input.LA(1);
					if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7516195840L) != 0)) {
						{
						setState(86);
						type_arg();
						setState(91);
						_errHandler.sync(this);
						_la = _input.LA(1);
						while (_la==T__5) {
							{
							{
							setState(87);
							match(T__5);
							setState(88);
							type_arg();
							}
							}
							setState(93);
							_errHandler.sync(this);
							_la = _input.LA(1);
						}
						}
					}

					setState(96);
					match(T__6);
					}
				}

				}
				break;
			case T__0:
				_localctx = new AnonTypeContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(99);
				match(T__0);
				setState(100);
				match(T__1);
				setState(104);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==IDENT) {
					{
					{
					setState(101);
					vardecl();
					}
					}
					setState(106);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(107);
				match(T__2);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Type_argContext extends ParserRuleContext {
		public Token elipsis;
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public Type_argContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_type_arg; }
	}

	public final Type_argContext type_arg() throws RecognitionException {
		Type_argContext _localctx = new Type_argContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_type_arg);
		try {
			setState(112);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__9:
				enterOuterAlt(_localctx, 1);
				{
				setState(110);
				((Type_argContext)_localctx).elipsis = match(T__9);
				}
				break;
			case T__10:
			case FLOAT:
			case INTEGER:
			case IDENT:
				enterOuterAlt(_localctx, 2);
				{
				setState(111);
				expr(0);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ConstantContext extends ParserRuleContext {
		public TerminalNode INTEGER() { return getToken(mdlParser.INTEGER, 0); }
		public TerminalNode FLOAT() { return getToken(mdlParser.FLOAT, 0); }
		public ConstantContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_constant; }
	}

	public final ConstantContext constant() throws RecognitionException {
		ConstantContext _localctx = new ConstantContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_constant);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(114);
			_la = _input.LA(1);
			if ( !(_la==FLOAT || _la==INTEGER) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class VardeclContext extends ParserRuleContext {
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public VardeclContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_vardecl; }
	}

	public final VardeclContext vardecl() throws RecognitionException {
		VardeclContext _localctx = new VardeclContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_vardecl);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(116);
			match(IDENT);
			setState(117);
			match(T__7);
			setState(118);
			type();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FuncdefContext extends ParserRuleContext {
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TemplatespecContext templatespec() {
			return getRuleContext(TemplatespecContext.class,0);
		}
		public List<ParamContext> param() {
			return getRuleContexts(ParamContext.class);
		}
		public ParamContext param(int i) {
			return getRuleContext(ParamContext.class,i);
		}
		public List<StatementContext> statement() {
			return getRuleContexts(StatementContext.class);
		}
		public StatementContext statement(int i) {
			return getRuleContext(StatementContext.class,i);
		}
		public FuncdefContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_funcdef; }
	}

	public final FuncdefContext funcdef() throws RecognitionException {
		FuncdefContext _localctx = new FuncdefContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_funcdef);
		int _la;
		try {
			setState(167);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,21,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(121);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__14) {
					{
					setState(120);
					templatespec();
					}
				}

				setState(123);
				type();
				setState(124);
				match(IDENT);
				setState(125);
				match(T__10);
				setState(134);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==IDENT) {
					{
					setState(126);
					param();
					setState(131);
					_errHandler.sync(this);
					_la = _input.LA(1);
					while (_la==T__5) {
						{
						{
						setState(127);
						match(T__5);
						setState(128);
						param();
						}
						}
						setState(133);
						_errHandler.sync(this);
						_la = _input.LA(1);
					}
					}
				}

				setState(136);
				match(T__11);
				setState(137);
				match(T__12);
				setState(138);
				expr(0);
				setState(139);
				match(T__13);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(142);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__14) {
					{
					setState(141);
					templatespec();
					}
				}

				setState(144);
				type();
				setState(145);
				match(IDENT);
				setState(146);
				match(T__10);
				setState(155);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==IDENT) {
					{
					setState(147);
					param();
					setState(152);
					_errHandler.sync(this);
					_la = _input.LA(1);
					while (_la==T__5) {
						{
						{
						setState(148);
						match(T__5);
						setState(149);
						param();
						}
						}
						setState(154);
						_errHandler.sync(this);
						_la = _input.LA(1);
					}
					}
				}

				setState(157);
				match(T__11);
				setState(158);
				match(T__1);
				setState(162);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 8321501186L) != 0)) {
					{
					{
					setState(159);
					statement();
					}
					}
					setState(164);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(165);
				match(T__2);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class TemplatespecContext extends ParserRuleContext {
		public List<VardeclContext> vardecl() {
			return getRuleContexts(VardeclContext.class);
		}
		public VardeclContext vardecl(int i) {
			return getRuleContext(VardeclContext.class,i);
		}
		public TemplatespecContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_templatespec; }
	}

	public final TemplatespecContext templatespec() throws RecognitionException {
		TemplatespecContext _localctx = new TemplatespecContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_templatespec);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(169);
			match(T__14);
			setState(170);
			match(T__15);
			setState(174);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==IDENT) {
				{
				{
				setState(171);
				vardecl();
				}
				}
				setState(176);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(177);
			match(T__16);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EinopsContext extends ParserRuleContext {
		public List<Einops_spec_fullContext> einops_spec_full() {
			return getRuleContexts(Einops_spec_fullContext.class);
		}
		public Einops_spec_fullContext einops_spec_full(int i) {
			return getRuleContext(Einops_spec_fullContext.class,i);
		}
		public List<KeyvalueContext> keyvalue() {
			return getRuleContexts(KeyvalueContext.class);
		}
		public KeyvalueContext keyvalue(int i) {
			return getRuleContext(KeyvalueContext.class,i);
		}
		public EinopsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_einops; }
	}

	public final EinopsContext einops() throws RecognitionException {
		EinopsContext _localctx = new EinopsContext(_ctx, getState());
		enterRule(_localctx, 20, RULE_einops);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(179);
			einops_spec_full();
			setState(180);
			match(T__17);
			setState(181);
			einops_spec_full();
			setState(193);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__18) {
				{
				setState(182);
				match(T__18);
				setState(191);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==IDENT) {
					{
					setState(183);
					keyvalue();
					setState(188);
					_errHandler.sync(this);
					_la = _input.LA(1);
					while (_la==T__5) {
						{
						{
						setState(184);
						match(T__5);
						setState(185);
						keyvalue();
						}
						}
						setState(190);
						_errHandler.sync(this);
						_la = _input.LA(1);
					}
					}
				}

				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Einops_spec_fullContext extends ParserRuleContext {
		public List<Einops_atomContext> einops_atom() {
			return getRuleContexts(Einops_atomContext.class);
		}
		public Einops_atomContext einops_atom(int i) {
			return getRuleContext(Einops_atomContext.class,i);
		}
		public Einops_spec_fullContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_einops_spec_full; }
	}

	public final Einops_spec_fullContext einops_spec_full() throws RecognitionException {
		Einops_spec_fullContext _localctx = new Einops_spec_fullContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_einops_spec_full);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(204); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				setState(204);
				_errHandler.sync(this);
				switch (_input.LA(1)) {
				case T__9:
				case T__19:
				case T__20:
				case INTEGER:
				case IDENT:
					{
					setState(195);
					einops_atom();
					}
					break;
				case T__10:
					{
					setState(196);
					match(T__10);
					setState(198); 
					_errHandler.sync(this);
					_la = _input.LA(1);
					do {
						{
						{
						setState(197);
						einops_atom();
						}
						}
						setState(200); 
						_errHandler.sync(this);
						_la = _input.LA(1);
					} while ( (((_la) & ~0x3f) == 0 && ((1L << _la) & 6445597696L) != 0) );
					setState(202);
					match(T__11);
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				}
				setState(206); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( (((_la) & ~0x3f) == 0 && ((1L << _la) & 6445599744L) != 0) );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Einops_atomContext extends ParserRuleContext {
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public TerminalNode INTEGER() { return getToken(mdlParser.INTEGER, 0); }
		public Einops_atomContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_einops_atom; }
	}

	public final Einops_atomContext einops_atom() throws RecognitionException {
		Einops_atomContext _localctx = new Einops_atomContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_einops_atom);
		try {
			setState(214);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case IDENT:
				enterOuterAlt(_localctx, 1);
				{
				setState(208);
				match(IDENT);
				}
				break;
			case T__9:
				enterOuterAlt(_localctx, 2);
				{
				setState(209);
				match(T__9);
				}
				break;
			case T__19:
				enterOuterAlt(_localctx, 3);
				{
				setState(210);
				match(T__19);
				}
				break;
			case INTEGER:
				enterOuterAlt(_localctx, 4);
				{
				setState(211);
				match(INTEGER);
				}
				break;
			case T__20:
				enterOuterAlt(_localctx, 5);
				{
				setState(212);
				match(T__20);
				setState(213);
				match(INTEGER);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class KeyvalueContext extends ParserRuleContext {
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public KeyvalueContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_keyvalue; }
	}

	public final KeyvalueContext keyvalue() throws RecognitionException {
		KeyvalueContext _localctx = new KeyvalueContext(_ctx, getState());
		enterRule(_localctx, 26, RULE_keyvalue);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(216);
			match(IDENT);
			setState(217);
			match(T__8);
			setState(218);
			expr(0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExprContext extends ParserRuleContext {
		public ExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr; }
	 
		public ExprContext() { }
		public void copyFrom(ExprContext ctx) {
			super.copyFrom(ctx);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class SwitchExpressionContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public SwitchExpressionContext(ExprContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ConstantExpressionContext extends ExprContext {
		public ConstantContext constant() {
			return getRuleContext(ConstantContext.class,0);
		}
		public ConstantExpressionContext(ExprContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ParenExpressionContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ParenExpressionContext(ExprContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class VarAccessExpressionContext extends ExprContext {
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public VarAccessExpressionContext(ExprContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class MethodcallExpressionContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public MethodcallExpressionContext(ExprContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ArithmeticExpressionContext extends ExprContext {
		public ExprContext left;
		public Token bop;
		public ExprContext right;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public ArithmeticExpressionContext(ExprContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class AccessExpressionContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public AccessExpressionContext(ExprContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class EinopsExpressionContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public EinopsContext einops() {
			return getRuleContext(EinopsContext.class,0);
		}
		public EinopsExpressionContext(ExprContext ctx) { copyFrom(ctx); }
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 28;
		enterRecursionRule(_localctx, 28, RULE_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(227);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case IDENT:
				{
				_localctx = new VarAccessExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(221);
				match(IDENT);
				}
				break;
			case FLOAT:
			case INTEGER:
				{
				_localctx = new ConstantExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(222);
				constant();
				}
				break;
			case T__10:
				{
				_localctx = new ParenExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(223);
				match(T__10);
				setState(224);
				expr(0);
				setState(225);
				match(T__11);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(269);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,35,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(267);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,34,_ctx) ) {
					case 1:
						{
						_localctx = new ArithmeticExpressionContext(new ExprContext(_parentctx, _parentState));
						((ArithmeticExpressionContext)_localctx).left = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(229);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(230);
						((ArithmeticExpressionContext)_localctx).bop = _input.LT(1);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 27262976L) != 0)) ) {
							((ArithmeticExpressionContext)_localctx).bop = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(231);
						((ArithmeticExpressionContext)_localctx).right = expr(4);
						}
						break;
					case 2:
						{
						_localctx = new ArithmeticExpressionContext(new ExprContext(_parentctx, _parentState));
						((ArithmeticExpressionContext)_localctx).left = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(232);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(233);
						((ArithmeticExpressionContext)_localctx).bop = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==T__24 || _la==T__25) ) {
							((ArithmeticExpressionContext)_localctx).bop = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(234);
						((ArithmeticExpressionContext)_localctx).right = expr(3);
						}
						break;
					case 3:
						{
						_localctx = new AccessExpressionContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(235);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(236);
						match(T__21);
						setState(237);
						match(IDENT);
						}
						break;
					case 4:
						{
						_localctx = new EinopsExpressionContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(238);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(239);
						match(T__4);
						setState(240);
						einops();
						setState(241);
						match(T__6);
						}
						break;
					case 5:
						{
						_localctx = new MethodcallExpressionContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(243);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(244);
						match(T__10);
						setState(253);
						_errHandler.sync(this);
						_la = _input.LA(1);
						if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7516194816L) != 0)) {
							{
							setState(245);
							expr(0);
							setState(250);
							_errHandler.sync(this);
							_la = _input.LA(1);
							while (_la==T__5) {
								{
								{
								setState(246);
								match(T__5);
								setState(247);
								expr(0);
								}
								}
								setState(252);
								_errHandler.sync(this);
								_la = _input.LA(1);
							}
							}
						}

						setState(255);
						match(T__11);
						}
						break;
					case 6:
						{
						_localctx = new SwitchExpressionContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(256);
						if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
						setState(257);
						match(T__26);
						setState(264);
						_errHandler.sync(this);
						_alt = getInterpreter().adaptivePredict(_input,33,_ctx);
						while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
							if ( _alt==1 ) {
								{
								{
								setState(258);
								expr(0);
								setState(259);
								match(T__12);
								setState(260);
								expr(0);
								}
								} 
							}
							setState(266);
							_errHandler.sync(this);
							_alt = getInterpreter().adaptivePredict(_input,33,_ctx);
						}
						}
						break;
					}
					} 
				}
				setState(271);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,35,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StatementContext extends ParserRuleContext {
		public StatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_statement; }
	 
		public StatementContext() { }
		public void copyFrom(StatementContext ctx) {
			super.copyFrom(ctx);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ExpressionStatementContext extends StatementContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ExpressionStatementContext(StatementContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ReturnStatementContext extends StatementContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ReturnStatementContext(StatementContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class DeclarationStatementContext extends StatementContext {
		public TerminalNode IDENT() { return getToken(mdlParser.IDENT, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public DeclarationStatementContext(StatementContext ctx) { copyFrom(ctx); }
	}

	public final StatementContext statement() throws RecognitionException {
		StatementContext _localctx = new StatementContext(_ctx, getState());
		enterRule(_localctx, 30, RULE_statement);
		try {
			setState(288);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,37,_ctx) ) {
			case 1:
				_localctx = new ExpressionStatementContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(272);
				expr(0);
				setState(273);
				match(T__13);
				}
				break;
			case 2:
				_localctx = new DeclarationStatementContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(277);
				_errHandler.sync(this);
				switch (_input.LA(1)) {
				case T__0:
				case IDENT:
					{
					setState(275);
					type();
					}
					break;
				case T__27:
					{
					setState(276);
					match(T__27);
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(279);
				match(IDENT);
				setState(280);
				match(T__8);
				setState(281);
				expr(0);
				setState(282);
				match(T__13);
				}
				break;
			case 3:
				_localctx = new ReturnStatementContext(_localctx);
				enterOuterAlt(_localctx, 3);
				{
				setState(284);
				match(T__28);
				setState(285);
				expr(0);
				setState(286);
				match(T__13);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 14:
			return expr_sempred((ExprContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 3);
		case 1:
			return precpred(_ctx, 2);
		case 2:
			return precpred(_ctx, 6);
		case 3:
			return precpred(_ctx, 5);
		case 4:
			return precpred(_ctx, 4);
		case 5:
			return precpred(_ctx, 1);
		}
		return true;
	}

	public static final String _serializedATN =
		"\u0004\u0001#\u0123\u0002\u0000\u0007\u0000\u0002\u0001\u0007\u0001\u0002"+
		"\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004\u0007\u0004\u0002"+
		"\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0002\u0007\u0007\u0007\u0002"+
		"\b\u0007\b\u0002\t\u0007\t\u0002\n\u0007\n\u0002\u000b\u0007\u000b\u0002"+
		"\f\u0007\f\u0002\r\u0007\r\u0002\u000e\u0007\u000e\u0002\u000f\u0007\u000f"+
		"\u0001\u0000\u0001\u0000\u0001\u0000\u0005\u0000$\b\u0000\n\u0000\f\u0000"+
		"\'\t\u0000\u0001\u0000\u0001\u0000\u0001\u0001\u0001\u0001\u0001\u0001"+
		"\u0001\u0001\u0005\u0001/\b\u0001\n\u0001\f\u00012\t\u0001\u0001\u0001"+
		"\u0001\u0001\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0005\u0002<\b\u0002\n\u0002\f\u0002?\t\u0002\u0003\u0002"+
		"A\b\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0005\u0002"+
		"G\b\u0002\n\u0002\f\u0002J\t\u0002\u0001\u0002\u0001\u0002\u0001\u0003"+
		"\u0001\u0003\u0001\u0003\u0001\u0003\u0001\u0003\u0003\u0003S\b\u0003"+
		"\u0001\u0004\u0001\u0004\u0001\u0004\u0001\u0004\u0001\u0004\u0005\u0004"+
		"Z\b\u0004\n\u0004\f\u0004]\t\u0004\u0003\u0004_\b\u0004\u0001\u0004\u0003"+
		"\u0004b\b\u0004\u0001\u0004\u0001\u0004\u0001\u0004\u0005\u0004g\b\u0004"+
		"\n\u0004\f\u0004j\t\u0004\u0001\u0004\u0003\u0004m\b\u0004\u0001\u0005"+
		"\u0001\u0005\u0003\u0005q\b\u0005\u0001\u0006\u0001\u0006\u0001\u0007"+
		"\u0001\u0007\u0001\u0007\u0001\u0007\u0001\b\u0003\bz\b\b\u0001\b\u0001"+
		"\b\u0001\b\u0001\b\u0001\b\u0001\b\u0005\b\u0082\b\b\n\b\f\b\u0085\t\b"+
		"\u0003\b\u0087\b\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0003"+
		"\b\u008f\b\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0005\b\u0097"+
		"\b\b\n\b\f\b\u009a\t\b\u0003\b\u009c\b\b\u0001\b\u0001\b\u0001\b\u0005"+
		"\b\u00a1\b\b\n\b\f\b\u00a4\t\b\u0001\b\u0001\b\u0003\b\u00a8\b\b\u0001"+
		"\t\u0001\t\u0001\t\u0005\t\u00ad\b\t\n\t\f\t\u00b0\t\t\u0001\t\u0001\t"+
		"\u0001\n\u0001\n\u0001\n\u0001\n\u0001\n\u0001\n\u0001\n\u0005\n\u00bb"+
		"\b\n\n\n\f\n\u00be\t\n\u0003\n\u00c0\b\n\u0003\n\u00c2\b\n\u0001\u000b"+
		"\u0001\u000b\u0001\u000b\u0004\u000b\u00c7\b\u000b\u000b\u000b\f\u000b"+
		"\u00c8\u0001\u000b\u0001\u000b\u0004\u000b\u00cd\b\u000b\u000b\u000b\f"+
		"\u000b\u00ce\u0001\f\u0001\f\u0001\f\u0001\f\u0001\f\u0001\f\u0003\f\u00d7"+
		"\b\f\u0001\r\u0001\r\u0001\r\u0001\r\u0001\u000e\u0001\u000e\u0001\u000e"+
		"\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0003\u000e\u00e4\b\u000e"+
		"\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e"+
		"\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e"+
		"\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e"+
		"\u0001\u000e\u0005\u000e\u00f9\b\u000e\n\u000e\f\u000e\u00fc\t\u000e\u0003"+
		"\u000e\u00fe\b\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001"+
		"\u000e\u0001\u000e\u0001\u000e\u0005\u000e\u0107\b\u000e\n\u000e\f\u000e"+
		"\u010a\t\u000e\u0005\u000e\u010c\b\u000e\n\u000e\f\u000e\u010f\t\u000e"+
		"\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0003\u000f"+
		"\u0116\b\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f"+
		"\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0003\u000f\u0121\b\u000f"+
		"\u0001\u000f\u0000\u0001\u001c\u0010\u0000\u0002\u0004\u0006\b\n\f\u000e"+
		"\u0010\u0012\u0014\u0016\u0018\u001a\u001c\u001e\u0000\u0003\u0001\u0000"+
		"\u001e\u001f\u0002\u0000\u0015\u0015\u0017\u0018\u0001\u0000\u0019\u001a"+
		"\u0142\u0000%\u0001\u0000\u0000\u0000\u0002*\u0001\u0000\u0000\u0000\u0004"+
		"5\u0001\u0000\u0000\u0000\u0006M\u0001\u0000\u0000\u0000\bl\u0001\u0000"+
		"\u0000\u0000\np\u0001\u0000\u0000\u0000\fr\u0001\u0000\u0000\u0000\u000e"+
		"t\u0001\u0000\u0000\u0000\u0010\u00a7\u0001\u0000\u0000\u0000\u0012\u00a9"+
		"\u0001\u0000\u0000\u0000\u0014\u00b3\u0001\u0000\u0000\u0000\u0016\u00cc"+
		"\u0001\u0000\u0000\u0000\u0018\u00d6\u0001\u0000\u0000\u0000\u001a\u00d8"+
		"\u0001\u0000\u0000\u0000\u001c\u00e3\u0001\u0000\u0000\u0000\u001e\u0120"+
		"\u0001\u0000\u0000\u0000 $\u0003\u0004\u0002\u0000!$\u0003\u0002\u0001"+
		"\u0000\"$\u0003\u0010\b\u0000# \u0001\u0000\u0000\u0000#!\u0001\u0000"+
		"\u0000\u0000#\"\u0001\u0000\u0000\u0000$\'\u0001\u0000\u0000\u0000%#\u0001"+
		"\u0000\u0000\u0000%&\u0001\u0000\u0000\u0000&(\u0001\u0000\u0000\u0000"+
		"\'%\u0001\u0000\u0000\u0000()\u0005\u0000\u0000\u0001)\u0001\u0001\u0000"+
		"\u0000\u0000*+\u0005\u0001\u0000\u0000+,\u0005 \u0000\u0000,0\u0005\u0002"+
		"\u0000\u0000-/\u0003\u000e\u0007\u0000.-\u0001\u0000\u0000\u0000/2\u0001"+
		"\u0000\u0000\u00000.\u0001\u0000\u0000\u000001\u0001\u0000\u0000\u0000"+
		"13\u0001\u0000\u0000\u000020\u0001\u0000\u0000\u000034\u0005\u0003\u0000"+
		"\u00004\u0003\u0001\u0000\u0000\u000056\u0005\u0004\u0000\u000067\u0005"+
		" \u0000\u00007@\u0005\u0005\u0000\u00008=\u0003\u0006\u0003\u00009:\u0005"+
		"\u0006\u0000\u0000:<\u0003\u0006\u0003\u0000;9\u0001\u0000\u0000\u0000"+
		"<?\u0001\u0000\u0000\u0000=;\u0001\u0000\u0000\u0000=>\u0001\u0000\u0000"+
		"\u0000>A\u0001\u0000\u0000\u0000?=\u0001\u0000\u0000\u0000@8\u0001\u0000"+
		"\u0000\u0000@A\u0001\u0000\u0000\u0000AB\u0001\u0000\u0000\u0000BC\u0005"+
		"\u0007\u0000\u0000CH\u0005\u0002\u0000\u0000DG\u0003\u000e\u0007\u0000"+
		"EG\u0003\u0010\b\u0000FD\u0001\u0000\u0000\u0000FE\u0001\u0000\u0000\u0000"+
		"GJ\u0001\u0000\u0000\u0000HF\u0001\u0000\u0000\u0000HI\u0001\u0000\u0000"+
		"\u0000IK\u0001\u0000\u0000\u0000JH\u0001\u0000\u0000\u0000KL\u0005\u0003"+
		"\u0000\u0000L\u0005\u0001\u0000\u0000\u0000MN\u0005 \u0000\u0000NO\u0005"+
		"\b\u0000\u0000OR\u0003\b\u0004\u0000PQ\u0005\t\u0000\u0000QS\u0003\u001c"+
		"\u000e\u0000RP\u0001\u0000\u0000\u0000RS\u0001\u0000\u0000\u0000S\u0007"+
		"\u0001\u0000\u0000\u0000Ta\u0005 \u0000\u0000U^\u0005\u0005\u0000\u0000"+
		"V[\u0003\n\u0005\u0000WX\u0005\u0006\u0000\u0000XZ\u0003\n\u0005\u0000"+
		"YW\u0001\u0000\u0000\u0000Z]\u0001\u0000\u0000\u0000[Y\u0001\u0000\u0000"+
		"\u0000[\\\u0001\u0000\u0000\u0000\\_\u0001\u0000\u0000\u0000][\u0001\u0000"+
		"\u0000\u0000^V\u0001\u0000\u0000\u0000^_\u0001\u0000\u0000\u0000_`\u0001"+
		"\u0000\u0000\u0000`b\u0005\u0007\u0000\u0000aU\u0001\u0000\u0000\u0000"+
		"ab\u0001\u0000\u0000\u0000bm\u0001\u0000\u0000\u0000cd\u0005\u0001\u0000"+
		"\u0000dh\u0005\u0002\u0000\u0000eg\u0003\u000e\u0007\u0000fe\u0001\u0000"+
		"\u0000\u0000gj\u0001\u0000\u0000\u0000hf\u0001\u0000\u0000\u0000hi\u0001"+
		"\u0000\u0000\u0000ik\u0001\u0000\u0000\u0000jh\u0001\u0000\u0000\u0000"+
		"km\u0005\u0003\u0000\u0000lT\u0001\u0000\u0000\u0000lc\u0001\u0000\u0000"+
		"\u0000m\t\u0001\u0000\u0000\u0000nq\u0005\n\u0000\u0000oq\u0003\u001c"+
		"\u000e\u0000pn\u0001\u0000\u0000\u0000po\u0001\u0000\u0000\u0000q\u000b"+
		"\u0001\u0000\u0000\u0000rs\u0007\u0000\u0000\u0000s\r\u0001\u0000\u0000"+
		"\u0000tu\u0005 \u0000\u0000uv\u0005\b\u0000\u0000vw\u0003\b\u0004\u0000"+
		"w\u000f\u0001\u0000\u0000\u0000xz\u0003\u0012\t\u0000yx\u0001\u0000\u0000"+
		"\u0000yz\u0001\u0000\u0000\u0000z{\u0001\u0000\u0000\u0000{|\u0003\b\u0004"+
		"\u0000|}\u0005 \u0000\u0000}\u0086\u0005\u000b\u0000\u0000~\u0083\u0003"+
		"\u0006\u0003\u0000\u007f\u0080\u0005\u0006\u0000\u0000\u0080\u0082\u0003"+
		"\u0006\u0003\u0000\u0081\u007f\u0001\u0000\u0000\u0000\u0082\u0085\u0001"+
		"\u0000\u0000\u0000\u0083\u0081\u0001\u0000\u0000\u0000\u0083\u0084\u0001"+
		"\u0000\u0000\u0000\u0084\u0087\u0001\u0000\u0000\u0000\u0085\u0083\u0001"+
		"\u0000\u0000\u0000\u0086~\u0001\u0000\u0000\u0000\u0086\u0087\u0001\u0000"+
		"\u0000\u0000\u0087\u0088\u0001\u0000\u0000\u0000\u0088\u0089\u0005\f\u0000"+
		"\u0000\u0089\u008a\u0005\r\u0000\u0000\u008a\u008b\u0003\u001c\u000e\u0000"+
		"\u008b\u008c\u0005\u000e\u0000\u0000\u008c\u00a8\u0001\u0000\u0000\u0000"+
		"\u008d\u008f\u0003\u0012\t\u0000\u008e\u008d\u0001\u0000\u0000\u0000\u008e"+
		"\u008f\u0001\u0000\u0000\u0000\u008f\u0090\u0001\u0000\u0000\u0000\u0090"+
		"\u0091\u0003\b\u0004\u0000\u0091\u0092\u0005 \u0000\u0000\u0092\u009b"+
		"\u0005\u000b\u0000\u0000\u0093\u0098\u0003\u0006\u0003\u0000\u0094\u0095"+
		"\u0005\u0006\u0000\u0000\u0095\u0097\u0003\u0006\u0003\u0000\u0096\u0094"+
		"\u0001\u0000\u0000\u0000\u0097\u009a\u0001\u0000\u0000\u0000\u0098\u0096"+
		"\u0001\u0000\u0000\u0000\u0098\u0099\u0001\u0000\u0000\u0000\u0099\u009c"+
		"\u0001\u0000\u0000\u0000\u009a\u0098\u0001\u0000\u0000\u0000\u009b\u0093"+
		"\u0001\u0000\u0000\u0000\u009b\u009c\u0001\u0000\u0000\u0000\u009c\u009d"+
		"\u0001\u0000\u0000\u0000\u009d\u009e\u0005\f\u0000\u0000\u009e\u00a2\u0005"+
		"\u0002\u0000\u0000\u009f\u00a1\u0003\u001e\u000f\u0000\u00a0\u009f\u0001"+
		"\u0000\u0000\u0000\u00a1\u00a4\u0001\u0000\u0000\u0000\u00a2\u00a0\u0001"+
		"\u0000\u0000\u0000\u00a2\u00a3\u0001\u0000\u0000\u0000\u00a3\u00a5\u0001"+
		"\u0000\u0000\u0000\u00a4\u00a2\u0001\u0000\u0000\u0000\u00a5\u00a6\u0005"+
		"\u0003\u0000\u0000\u00a6\u00a8\u0001\u0000\u0000\u0000\u00a7y\u0001\u0000"+
		"\u0000\u0000\u00a7\u008e\u0001\u0000\u0000\u0000\u00a8\u0011\u0001\u0000"+
		"\u0000\u0000\u00a9\u00aa\u0005\u000f\u0000\u0000\u00aa\u00ae\u0005\u0010"+
		"\u0000\u0000\u00ab\u00ad\u0003\u000e\u0007\u0000\u00ac\u00ab\u0001\u0000"+
		"\u0000\u0000\u00ad\u00b0\u0001\u0000\u0000\u0000\u00ae\u00ac\u0001\u0000"+
		"\u0000\u0000\u00ae\u00af\u0001\u0000\u0000\u0000\u00af\u00b1\u0001\u0000"+
		"\u0000\u0000\u00b0\u00ae\u0001\u0000\u0000\u0000\u00b1\u00b2\u0005\u0011"+
		"\u0000\u0000\u00b2\u0013\u0001\u0000\u0000\u0000\u00b3\u00b4\u0003\u0016"+
		"\u000b\u0000\u00b4\u00b5\u0005\u0012\u0000\u0000\u00b5\u00c1\u0003\u0016"+
		"\u000b\u0000\u00b6\u00bf\u0005\u0013\u0000\u0000\u00b7\u00bc\u0003\u001a"+
		"\r\u0000\u00b8\u00b9\u0005\u0006\u0000\u0000\u00b9\u00bb\u0003\u001a\r"+
		"\u0000\u00ba\u00b8\u0001\u0000\u0000\u0000\u00bb\u00be\u0001\u0000\u0000"+
		"\u0000\u00bc\u00ba\u0001\u0000\u0000\u0000\u00bc\u00bd\u0001\u0000\u0000"+
		"\u0000\u00bd\u00c0\u0001\u0000\u0000\u0000\u00be\u00bc\u0001\u0000\u0000"+
		"\u0000\u00bf\u00b7\u0001\u0000\u0000\u0000\u00bf\u00c0\u0001\u0000\u0000"+
		"\u0000\u00c0\u00c2\u0001\u0000\u0000\u0000\u00c1\u00b6\u0001\u0000\u0000"+
		"\u0000\u00c1\u00c2\u0001\u0000\u0000\u0000\u00c2\u0015\u0001\u0000\u0000"+
		"\u0000\u00c3\u00cd\u0003\u0018\f\u0000\u00c4\u00c6\u0005\u000b\u0000\u0000"+
		"\u00c5\u00c7\u0003\u0018\f\u0000\u00c6\u00c5\u0001\u0000\u0000\u0000\u00c7"+
		"\u00c8\u0001\u0000\u0000\u0000\u00c8\u00c6\u0001\u0000\u0000\u0000\u00c8"+
		"\u00c9\u0001\u0000\u0000\u0000\u00c9\u00ca\u0001\u0000\u0000\u0000\u00ca"+
		"\u00cb\u0005\f\u0000\u0000\u00cb\u00cd\u0001\u0000\u0000\u0000\u00cc\u00c3"+
		"\u0001\u0000\u0000\u0000\u00cc\u00c4\u0001\u0000\u0000\u0000\u00cd\u00ce"+
		"\u0001\u0000\u0000\u0000\u00ce\u00cc\u0001\u0000\u0000\u0000\u00ce\u00cf"+
		"\u0001\u0000\u0000\u0000\u00cf\u0017\u0001\u0000\u0000\u0000\u00d0\u00d7"+
		"\u0005 \u0000\u0000\u00d1\u00d7\u0005\n\u0000\u0000\u00d2\u00d7\u0005"+
		"\u0014\u0000\u0000\u00d3\u00d7\u0005\u001f\u0000\u0000\u00d4\u00d5\u0005"+
		"\u0015\u0000\u0000\u00d5\u00d7\u0005\u001f\u0000\u0000\u00d6\u00d0\u0001"+
		"\u0000\u0000\u0000\u00d6\u00d1\u0001\u0000\u0000\u0000\u00d6\u00d2\u0001"+
		"\u0000\u0000\u0000\u00d6\u00d3\u0001\u0000\u0000\u0000\u00d6\u00d4\u0001"+
		"\u0000\u0000\u0000\u00d7\u0019\u0001\u0000\u0000\u0000\u00d8\u00d9\u0005"+
		" \u0000\u0000\u00d9\u00da\u0005\t\u0000\u0000\u00da\u00db\u0003\u001c"+
		"\u000e\u0000\u00db\u001b\u0001\u0000\u0000\u0000\u00dc\u00dd\u0006\u000e"+
		"\uffff\uffff\u0000\u00dd\u00e4\u0005 \u0000\u0000\u00de\u00e4\u0003\f"+
		"\u0006\u0000\u00df\u00e0\u0005\u000b\u0000\u0000\u00e0\u00e1\u0003\u001c"+
		"\u000e\u0000\u00e1\u00e2\u0005\f\u0000\u0000\u00e2\u00e4\u0001\u0000\u0000"+
		"\u0000\u00e3\u00dc\u0001\u0000\u0000\u0000\u00e3\u00de\u0001\u0000\u0000"+
		"\u0000\u00e3\u00df\u0001\u0000\u0000\u0000\u00e4\u010d\u0001\u0000\u0000"+
		"\u0000\u00e5\u00e6\n\u0003\u0000\u0000\u00e6\u00e7\u0007\u0001\u0000\u0000"+
		"\u00e7\u010c\u0003\u001c\u000e\u0004\u00e8\u00e9\n\u0002\u0000\u0000\u00e9"+
		"\u00ea\u0007\u0002\u0000\u0000\u00ea\u010c\u0003\u001c\u000e\u0003\u00eb"+
		"\u00ec\n\u0006\u0000\u0000\u00ec\u00ed\u0005\u0016\u0000\u0000\u00ed\u010c"+
		"\u0005 \u0000\u0000\u00ee\u00ef\n\u0005\u0000\u0000\u00ef\u00f0\u0005"+
		"\u0005\u0000\u0000\u00f0\u00f1\u0003\u0014\n\u0000\u00f1\u00f2\u0005\u0007"+
		"\u0000\u0000\u00f2\u010c\u0001\u0000\u0000\u0000\u00f3\u00f4\n\u0004\u0000"+
		"\u0000\u00f4\u00fd\u0005\u000b\u0000\u0000\u00f5\u00fa\u0003\u001c\u000e"+
		"\u0000\u00f6\u00f7\u0005\u0006\u0000\u0000\u00f7\u00f9\u0003\u001c\u000e"+
		"\u0000\u00f8\u00f6\u0001\u0000\u0000\u0000\u00f9\u00fc\u0001\u0000\u0000"+
		"\u0000\u00fa\u00f8\u0001\u0000\u0000\u0000\u00fa\u00fb\u0001\u0000\u0000"+
		"\u0000\u00fb\u00fe\u0001\u0000\u0000\u0000\u00fc\u00fa\u0001\u0000\u0000"+
		"\u0000\u00fd\u00f5\u0001\u0000\u0000\u0000\u00fd\u00fe\u0001\u0000\u0000"+
		"\u0000\u00fe\u00ff\u0001\u0000\u0000\u0000\u00ff\u010c\u0005\f\u0000\u0000"+
		"\u0100\u0101\n\u0001\u0000\u0000\u0101\u0108\u0005\u001b\u0000\u0000\u0102"+
		"\u0103\u0003\u001c\u000e\u0000\u0103\u0104\u0005\r\u0000\u0000\u0104\u0105"+
		"\u0003\u001c\u000e\u0000\u0105\u0107\u0001\u0000\u0000\u0000\u0106\u0102"+
		"\u0001\u0000\u0000\u0000\u0107\u010a\u0001\u0000\u0000\u0000\u0108\u0106"+
		"\u0001\u0000\u0000\u0000\u0108\u0109\u0001\u0000\u0000\u0000\u0109\u010c"+
		"\u0001\u0000\u0000\u0000\u010a\u0108\u0001\u0000\u0000\u0000\u010b\u00e5"+
		"\u0001\u0000\u0000\u0000\u010b\u00e8\u0001\u0000\u0000\u0000\u010b\u00eb"+
		"\u0001\u0000\u0000\u0000\u010b\u00ee\u0001\u0000\u0000\u0000\u010b\u00f3"+
		"\u0001\u0000\u0000\u0000\u010b\u0100\u0001\u0000\u0000\u0000\u010c\u010f"+
		"\u0001\u0000\u0000\u0000\u010d\u010b\u0001\u0000\u0000\u0000\u010d\u010e"+
		"\u0001\u0000\u0000\u0000\u010e\u001d\u0001\u0000\u0000\u0000\u010f\u010d"+
		"\u0001\u0000\u0000\u0000\u0110\u0111\u0003\u001c\u000e\u0000\u0111\u0112"+
		"\u0005\u000e\u0000\u0000\u0112\u0121\u0001\u0000\u0000\u0000\u0113\u0116"+
		"\u0003\b\u0004\u0000\u0114\u0116\u0005\u001c\u0000\u0000\u0115\u0113\u0001"+
		"\u0000\u0000\u0000\u0115\u0114\u0001\u0000\u0000\u0000\u0116\u0117\u0001"+
		"\u0000\u0000\u0000\u0117\u0118\u0005 \u0000\u0000\u0118\u0119\u0005\t"+
		"\u0000\u0000\u0119\u011a\u0003\u001c\u000e\u0000\u011a\u011b\u0005\u000e"+
		"\u0000\u0000\u011b\u0121\u0001\u0000\u0000\u0000\u011c\u011d\u0005\u001d"+
		"\u0000\u0000\u011d\u011e\u0003\u001c\u000e\u0000\u011e\u011f\u0005\u000e"+
		"\u0000\u0000\u011f\u0121\u0001\u0000\u0000\u0000\u0120\u0110\u0001\u0000"+
		"\u0000\u0000\u0120\u0115\u0001\u0000\u0000\u0000\u0120\u011c\u0001\u0000"+
		"\u0000\u0000\u0121\u001f\u0001\u0000\u0000\u0000&#%0=@FHR[^ahlpy\u0083"+
		"\u0086\u008e\u0098\u009b\u00a2\u00a7\u00ae\u00bc\u00bf\u00c1\u00c8\u00cc"+
		"\u00ce\u00d6\u00e3\u00fa\u00fd\u0108\u010b\u010d\u0115\u0120";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}