!function(){"use strict";var t,e,a,u,n={667:function(t,e){e.q=void 0,e.q="4.0.0-alpha.1"},738:function(t,e,a){var u=this&&this.__importDefault||function(t){return t&&t.__esModule?t:{default:t}};Object.defineProperty(e,"__esModule",{value:!0}),e.HtmlConfiguration=void 0;var n=a(251),r=a(871),o=u(a(248));new r.CommandMap("html_macros",{data:"Data",href:"Href",class:"Class",style:"Style",cssId:"Id"},o.default),e.HtmlConfiguration=n.Configuration.create("html",{handler:{macro:["html_macros"]}})},248:function(t,e,a){var u=this&&this.__importDefault||function(t){return t&&t.__esModule?t:{default:t}};Object.defineProperty(e,"__esModule",{value:!0});var n=u(a(748)),r=u(a(398)),o=u(a(402)),F={Data:function(t,e){var a=t.GetArgument(e),u=f(t,e),F=r.default.keyvalOptions(a);for(var i in F){if(!l(i))throw new o.default("InvalidHTMLAttr","Invalid HTML attribute: %1","data-".concat(i));n.default.setAttribute(u,"data-".concat(i),F[i])}t.Push(u)}},i=/[\u{FDD0}-\u{FDEF}\u{FFFE}\u{FFFF}\u{1FFFE}\u{1FFFF}\u{2FFFE}\u{2FFFF}\u{3FFFE}\u{3FFFF}\u{4FFFE}\u{4FFFF}\u{5FFFE}\u{5FFFF}\u{6FFFE}\u{6FFFF}\u{7FFFE}\u{7FFFF}\u{8FFFE}\u{8FFFF}\u{9FFFE}\u{9FFFF}\u{AFFFE}\u{AFFFF}\u{BFFFE}\u{BFFFF}\u{CFFFE}\u{CFFFF}\u{DFFFE}\u{DFFFF}\u{EFFFE}\u{EFFFF}\u{FFFFE}\u{FFFFF}\u{10FFFE}\u{10FFFF}]/u;function l(t){return!(t.match(/[\x00-\x1f\x7f-\x9f "'>\/=]/)||t.match(i))}F.Href=function(t,e){var a=t.GetArgument(e),u=f(t,e);n.default.setAttribute(u,"href",a),t.Push(u)},F.Class=function(t,e){var a=t.GetArgument(e),u=f(t,e),r=n.default.getAttribute(u,"class");r&&(a=r+" "+a),n.default.setAttribute(u,"class",a),t.Push(u)},F.Style=function(t,e){var a=t.GetArgument(e),u=f(t,e),r=n.default.getAttribute(u,"style");r&&(";"!==a.charAt(a.length-1)&&(a+=";"),a=r+" "+a),n.default.setAttribute(u,"style",a),t.Push(u)},F.Id=function(t,e){var a=t.GetArgument(e),u=f(t,e);n.default.setAttribute(u,"id",a),t.Push(u)};var f=function(t,e){var a=t.ParseArg(e);if(!n.default.isInferred(a))return a;var u=n.default.getChildren(a);if(1===u.length)return u[0];var r=t.create("node","mrow");return n.default.copyChildren(a,r),n.default.copyAttributes(a,r),r};e.default=F},955:function(t,e){MathJax._.components.global.isObject,MathJax._.components.global.combineConfig,MathJax._.components.global.combineDefaults,e.r8=MathJax._.components.global.combineWithMathJax,MathJax._.components.global.MathJax},251:function(t,e){Object.defineProperty(e,"__esModule",{value:!0}),e.Configuration=MathJax._.input.tex.Configuration.Configuration,e.ConfigurationHandler=MathJax._.input.tex.Configuration.ConfigurationHandler,e.ParserConfiguration=MathJax._.input.tex.Configuration.ParserConfiguration},748:function(t,e){Object.defineProperty(e,"__esModule",{value:!0}),e.default=MathJax._.input.tex.NodeUtil.default},398:function(t,e){Object.defineProperty(e,"__esModule",{value:!0}),e.default=MathJax._.input.tex.ParseUtil.default},871:function(t,e){Object.defineProperty(e,"__esModule",{value:!0}),e.parseResult=MathJax._.input.tex.SymbolMap.parseResult,e.AbstractSymbolMap=MathJax._.input.tex.SymbolMap.AbstractSymbolMap,e.RegExpMap=MathJax._.input.tex.SymbolMap.RegExpMap,e.AbstractParseMap=MathJax._.input.tex.SymbolMap.AbstractParseMap,e.CharacterMap=MathJax._.input.tex.SymbolMap.CharacterMap,e.DelimiterMap=MathJax._.input.tex.SymbolMap.DelimiterMap,e.MacroMap=MathJax._.input.tex.SymbolMap.MacroMap,e.CommandMap=MathJax._.input.tex.SymbolMap.CommandMap,e.EnvironmentMap=MathJax._.input.tex.SymbolMap.EnvironmentMap},402:function(t,e){Object.defineProperty(e,"__esModule",{value:!0}),e.default=MathJax._.input.tex.TexError.default}},r={};function o(t){var e=r[t];if(void 0!==e)return e.exports;var a=r[t]={exports:{}};return n[t].call(a.exports,a,a.exports,o),a.exports}t=o(955),e=o(667),a=o(738),u=o(248),MathJax.loader&&MathJax.loader.checkVersion("[tex]/html",e.q,"tex-extension"),(0,t.r8)({_:{input:{tex:{html:{HtmlConfiguration:a,HtmlMethods:u}}}}})}();