!function(){"use strict";var t,o,a,n={667:function(t,o){o.q=void 0,o.q="4.0.0-alpha.1"},941:function(t,o,a){var n,r=this&&this.__extends||(n=function(t,o){return n=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(t,o){t.__proto__=o}||function(t,o){for(var a in o)Object.prototype.hasOwnProperty.call(o,a)&&(t[a]=o[a])},n(t,o)},function(t,o){if("function"!=typeof o&&null!==o)throw new TypeError("Class extends value "+String(o)+" is not a constructor or null");function a(){this.constructor=t}n(t,o),t.prototype=null===o?Object.create(o):(a.prototype=o.prototype,new a)});Object.defineProperty(o,"__esModule",{value:!0}),o.TagFormatConfiguration=o.tagformatConfig=void 0;var e=a(251),i=a(680),s=0;function u(t,o){var a=o.parseOptions.options.tags;"base"!==a&&t.tags.hasOwnProperty(a)&&i.TagsFactory.add(a,t.tags[a]);var n=function(t){function a(){return null!==t&&t.apply(this,arguments)||this}return r(a,t),a.prototype.formatNumber=function(t){return o.parseOptions.options.tagformat.number(t)},a.prototype.formatTag=function(t){return o.parseOptions.options.tagformat.tag(t)},a.prototype.formatId=function(t){return o.parseOptions.options.tagformat.id(t)},a.prototype.formatUrl=function(t,a){return o.parseOptions.options.tagformat.url(t,a)},a}(i.TagsFactory.create(o.parseOptions.options.tags).constructor),e="configTags-"+ ++s;i.TagsFactory.add(e,n),o.parseOptions.options.tags=e}o.tagformatConfig=u,o.TagFormatConfiguration=e.Configuration.create("tagformat",{config:[u,10],options:{tagformat:{number:function(t){return t.toString()},tag:function(t){return"("+t+")"},id:function(t){return"mjx-eqn:"+t.replace(/\s/g,"_")},url:function(t,o){return o+"#"+encodeURIComponent(t)}}}})},955:function(t,o){MathJax._.components.global.isObject,MathJax._.components.global.combineConfig,MathJax._.components.global.combineDefaults,o.r8=MathJax._.components.global.combineWithMathJax,MathJax._.components.global.MathJax},251:function(t,o){Object.defineProperty(o,"__esModule",{value:!0}),o.Configuration=MathJax._.input.tex.Configuration.Configuration,o.ConfigurationHandler=MathJax._.input.tex.Configuration.ConfigurationHandler,o.ParserConfiguration=MathJax._.input.tex.Configuration.ParserConfiguration},680:function(t,o){Object.defineProperty(o,"__esModule",{value:!0}),o.Label=MathJax._.input.tex.Tags.Label,o.TagInfo=MathJax._.input.tex.Tags.TagInfo,o.AbstractTags=MathJax._.input.tex.Tags.AbstractTags,o.NoTags=MathJax._.input.tex.Tags.NoTags,o.AllTags=MathJax._.input.tex.Tags.AllTags,o.TagsFactory=MathJax._.input.tex.Tags.TagsFactory}},r={};function e(t){var o=r[t];if(void 0!==o)return o.exports;var a=r[t]={exports:{}};return n[t].call(a.exports,a,a.exports,e),a.exports}t=e(955),o=e(667),a=e(941),MathJax.loader&&MathJax.loader.checkVersion("[tex]/tagformat",o.q,"tex-extension"),(0,t.r8)({_:{input:{tex:{tagformat:{TagFormatConfiguration:a}}}}})}();