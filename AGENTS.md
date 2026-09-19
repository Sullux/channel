# Agents

Remember these principles as you engage in coding tasks.

## Core Principles

- Minimal dependencies. Sometimes, third-party libraries are indeed necessary; however, we will always look at a home-rolled solution first.
- Trustworthy dependencies. We will not add a third-party dependency without reviewing it and affirming our trust in it.
- Reliable dependencies. We will not add a third-party dependency without an exact, unchanging binary version and/or code base.

# Coding Guidelines

## Zig

Try to keep code files short. While 100 lines might be an extremely tight limitation in a language like Zig, if a file passes 200 lines then it is probably time to think about refactoring it if possible.

Try to minimize indentation and nesting where reasonable. Early function return is preferable to `else` clauses.

## JavaScript

- Vanilla JavaScript only: no TypeScript, minimal dependencies, minimal build steps (or minimal copy/paste)
  - Note: Do not add dependencies for libraries that do simple jobs. Better to write our own implementation for e.g. ULIDs, loggers, etc. Only add dependencies where absolutely necessary
- Code must be auditable and readable: all dependencies visible, no transpilation
- Functional programming style: factories over classes, const over let, comprehensions over loops, ternary over branching (except performance-critical DB ops)
- No classes, no `this`
- Code style is prettier.js but ADD trailing commas and REMOVE semicolons (see .prettierrc for details)
- Prefer CJS for backend, MJS for frontend but defer to local project constraints
- Use Node.js built-in testing framework instead of a 3rd-party framework
- Use `yarn` instead of `npm`
- Naming: Always PascalCase (factories), SHOUT_CASE (app-level or module-level constants) and camelCase (everything else)
  - GOOD: `const bucketPrefix = foo` (camelCase variable)
  - GOOD: `const BucketDb = (config) => {...}` (PascalCase factory)
  - BAD: `const bucket_prefix = foo` (snake_case variable)
  - BAD: `function BucketDb (config) {...}` (Why bad `DB` instead of good `Db`?)
- For readability, lines should be 80 characters or shorter where reasonable
- for imports/requires, do not include the file extension unless necessary
  - GOOD: `const { bar, baz } = require('./foo')`
  - BAD: `const { bar, baz } = require('./foo.js')`
- Bias towards shorter code files no longer than 100 lines while maintaining readability
  - Make exceptions where reasonable e.g. a translation layer full of simple A -> B mapping functions is not reasonable to split up
  - Don't be afraid to turn a long code file into a folder with an `index.js` pulling from multiple sub-files e.g. `foo.js` becomes `foo/index.js`, `foo/bar.js` and `foo/baz.js` so as not to break existing imports (because `require(./foo)` still works before and after)
- for side effects, name functions for what they do e.g. `saveFile()` or `queryApi()`
- for pure functions, name functions for what they return:
  - GOOD: `const normalizedEmail = (email) => email.toLowerCase()` (function name describes the return value)
  - BAD: `const normalizeEmail = (email) => email.toLowerCase()` (function name describes an action: incorrect!)

ALWAYS use dependency injection for non-deterministic dependencies. Example:

```javascript
// BAD
export const Widget = (x, y) => ({
  x,
  y,
  timestamp: Date.now(),
}) // a factory that produces widgets

// GOOD
export const widgetFactory = (now) => (x, y) => ({
  x,
  y,
  timestamp: now(),
}) // a dependency injection wrapper function that returns a Widget factory
```

By adding a dependency injection layer, a unit test can pass a simple mock e.g.

```javascript
import { widgetFactory } from './widget'
// later, in test code:
const Widget = widgetFactory(() => 1)
// etc.
```

And in the local `index.js` file, it can be exported with its runtime dependencies:

```javascript
import { widgetFactory } from './widget'
export const Widget = widgetFactory(Date.now)
```

This pattern is especially helpful to ensure that side effects e.g. logging can be safely tested while keeping the test output uncluttered.

Special note about naming: the earlier rules state, "Always PascalCase (factories)". The above appears to violate that with `widgetFactory`; however, there is another rule: "for pure functions, name functions for what they return". In the above example, the dependency injection wrapper function _returns a Widget factory_; thus the name `widgetFactory` is correct, and it returns a factory function correctly named `Widget` (PascalCase).

# REMEMBER

If you do not see any of these files in your context, be sure to read them in their entirety before doing any work:

- `PROBLEM.md`
- `APPROACH.md`
- `ARCHITECTURE.md`
- `MODEL_SELECTION.md`
- `ENGINEERING.md`

## Current Project Goals

- Achieve the same or nearly the same performance as llama.cpp, including ~ 2 second prompt processing and ~ 27 tok/s generation speed at Q4_0 quantization.
- Do not deviate from llama.cpp's approach in order to ensure speed and correctness.
- Do not deviate from the python reference implementation in order to ensure correctness with the Gemma 4 family of models.
- Keep our `ENGINEERING.md` document in mind so that we do not accidentally repeat a failed experiment or break a necessary part of our system.
