# Merlin Design Document

## Modules

Merlin adopts `tf.Module` as the core building block rather than Keras layers (which themselves inherit from `tf.Module`). This is to minimize the amount of "magic" that happens and keep things as simple as possible.

Merlin provides a subclass of `tf.Module` that provides certain additional features (such as auto-generating unique names as described below). However, this subclass as entirely optional and other parts of Merlin are largely agnostic to it and will work just fine with any `tf.Module` subclass (or indeed in most cases, any callable).

### Naming

Merlin's `Module` subclass auto-generates unique names. While this is not strictly required in TensorFlow 2.0, lack of unique superficial names can be quite confusing in practice.

-   Names are unique within the construction namescope
-   If no explicit name is provided, one is generated from the type name
-   Colliding names are resolved by suffixing them with an increasing index, starting at 1.

### Scoping Issues

The actual implementation of `tf.Module`, unlike the [original RFC for modules in TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20190117-tf-module.md), does not automatically nest its submodules within its namescope. Implementing auto-scoping as described in the RFC is fraught with gotchas. To demonstrate some of these issues, consider the following module:

```python
class Contrived(tf.Module):
    def __init__(self):
        self.alpha = tf.Variable(1.0, name='alpha')
        self.beta = None

    def __call__(self, x):
        if self.beta is None:
            self.beta = tf.Variable(x, name='beta')
        return self.alpha + x * self.beta
```

There are two points in time where a variable is created: once during construction and the other during invocation. Both of these scenarios are common. Unfortunately, nothing enforces the enclosing namescope to be the same at both points. Inconsistent scoping is just a step away. There are a few possible options for auto-scoping:

-   The approach used by DeepMind's Sonnet: the base class' `__init__` enters the namescope and leaves it dangling for the subclass. A custom metaclass later exits this scope. It goes a step further and wraps other methods within scopes as well. While this approach does result in the desired behavior, it introduces a fair amount of complexity. Furthermore, it relies on the subclasses invoking `super` before any variable creation.

-   An approach used initially while prototyping Merlin but subsequently abandoned: using `__init__subclass__` to wrap the inherited subclass' constructor that ensures the namescope is entered. This also allows inheritance-time arguments to opt-out of this behavior:

    ```python
    class Contrived(merlin.Module, autoscope=False):
        ...
    ```

    Unfortunately, this approach requires that the decorator somehow extract the desired name. The prototype established a convention of using a keyword argument `name` for this. While this works in the common case, it quickly breaks down (for instance, when a subclass `__init__` accepts a `Spec` instance that encapsulates the name).

Furthermore, neither of these solutions work around many other scoping edge cases. Consider a `Sequential` module (as described in the RFC). Constructing it as follows:

```python
unit = Sequential([Contrived(), Contrived()], name='outer')
```

fails to nest any of the submodules within the parents namescope since they are constructed before the `Sequential` instance. Even if you use the scoping mechanism provided by Merlin's `Sequential` implementation, scoping can still be subverted:

```python
with Sequential(name='outer') as outer:
    special = Contrived()
    outer += [special, Contrived()]
```

While both submodules are now constructed within `outer`'s scope, the variable names of `special` depend on which is invoked first: `special` or `outer`.

Merlin opts for simplicity and explicitness here. The core `Module` class implements no scoping magic for `__init__`. Instead, commonly used convenience subclasses (such as those provided for use with `Spec` based configurations) use simple indirection for both construction and invocation (eg: see `merlin.modules.configurable.Composite`).

## The `Spec` class

Desirables:

-   A way to concisely describe hyperparameters/options for various components (modules, trainers, etc)
-   Type annotations for fields
-   Safe defaults (no accidental mutation causing changes to past/future instances)
-   Validated (typos shouldn't result in new fields, must provided non-default fields)
-   Composable and inheritable
-   Suitable for serialization/deserialization to/from JSON

Python 3's `dataclasses` meet many of these requirements (and, consequently, forms the core of the `Spec` class) but fall short on some fronts. In particular:

-   Subclasses inheriting from dataclass decorated parents must be explicitly decorated again. (`Spec` addresses this by auto-transforming subclasses to dataclasses.)

-   Fields missing type annotations are silently ignored. (`Spec` asserts all fields have type annotations.)

-   Inheriting dataclasses with defaults with a custom init (as desired) results. (`Spec` works around this by implementing a custom init and requiring keyword-only arguments.)

In addition, `Spec` provides additional utilities (eg: a `Mapping` interface).

Within Merlin, a common idiom is to use a nested `Spec` subclass (often named `Config`) for specifying the class' hyper parameters. Since a `Spec` instance is a valid `Mapping`, the following allows it to co-exist with traditional keyword arguments:

```python
conv = Conv2D(filters=256, kernel_size=3)

config = Conv2D.Config(filters=256, kernel_size=3)
...
conv = Conv2D(**config)
```

## Consistency

The introduction of Keras into TensorFlow core has lead to a few inconsistencies. For instance, core TensorFlow operations refer to data orderings using constants like `nchw` and `nhwc`, whereas Keras uses `channels_first` and `channels_last`.

All modules in Merlin use the TensorFlow core constants whenever possible. When wrapping Keras layers (such as `Conv2D`), the constants are automatically translated.

## Explicitness

There may be trade-offs between eliminating boilerplate code at the cost of making things less explicit. For instance, [tf-slim](https://ai.googleblog.com/2016/08/tf-slim-high-level-library-to-define.html) allows using "arg scopes" to eliminate repeated common arguments (activation/normalization functions, initializers, etc). While this solves the issue of repeatedly specifying these arguments everywhere, it can make it hard to understand the code later (is this just a Conv2D, or is there a ReLU + BatchNorm auto-injected by an `arg_scope` a few levels above? Does this function expect the caller to nest it within a scope to function correctly? Did the `arg_scope` inadvertently inject an activation on top of the logits?).

Merlin aims to err on the side of explicitness. Boilerplate reduction is approached via constructs like `NormalizingActivation` which aim to concisely yet explicitly capture commonly used idioms.

## Future Development

While Merlin is usable for non-trivial projects (eg: DeepLab v3), it's still missing a number of features (such as distributed training). Some of these are best deferred until after TensorFlow 2.0 is released.
