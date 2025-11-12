### Polars DataFrame Head Example

Source: https://docs.rs/polars/latest/polars/frame/struct

Demonstrates how to use the `head` method to get the first few rows of a DataFrame.

```rust
let countries: DataFrame = df!("Rank by GDP (2021)" => [1, 2, 3, 4, 5],
        "Continent" => ["North America", "Asia", "Asia", "Europe", "Europe"],
        "Country" => ["United States", "China", "Japan", "Germany", "United Kingdom"],
        "Capital" => ["Washington", "Beijing", "Tokyo", "Berlin", "London"])?;
assert_eq!(countries.shape(), (5, 4));

println!("{}", countries.head(Some(3)));
```

---

### Rust Documentation Resources

Source: https://docs.rs/polars/latest/polars/testing/index

Provides links to essential Rust documentation resources, including the official Rust website, The Book, standard library API reference, Rust by Example, and the Cargo Guide.

```Rust
* [Rust website](https://www.rust-lang.org/)
    * [The Book](https://doc.rust-lang.org/book/)
    * [Standard Library API Reference](https://doc.rust-lang.org/std/)
    * [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
    * [The Cargo Guide](https://doc.rust-lang.org/cargo/guide/)
    * [Clippy Documentation](https://doc.rust-lang.org/nightly/clippy)
```

---

### Polars Module Documentation

Source: https://docs.rs/polars/latest/polars/prelude/mkdir/index

Provides links to the Polars documentation, specifically highlighting the 'mkdir' module and the main Polars API. It also includes links to the Rust website, The Book, Standard Library API Reference, Rust by Example, Cargo Guide, and Clippy Documentation for broader Rust development context.

```Rust
https://docs.rs/polars/latest/polars/prelude/mkdir/index.html
https://docs.rs/polars/latest/polars/index.html
https://www.rust-lang.org/
https://doc.rust-lang.org/book/
https://doc.rust-lang.org/std/
https://doc.rust-lang.org/rust-by-example/
https://doc.rust-lang.org/cargo/guide/
https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/chunked_array/ops/fill_null/index

Offers links to essential Rust resources, including the official Rust website, The Book, standard library API reference, Rust by Example, and the Cargo Guide.

```Rust
rust_resources:
  - website: https://www.rust-lang.org/
  - book: https://doc.rust-lang.org/book/
  - std_api: https://doc.rust-lang.org/std/
  - by_example: https://doc.rust-lang.org/rust-by-example/
  - cargo_guide: https://doc.rust-lang.org/cargo/guide/
  - clippy: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/prelude/binary/index

Provides links to essential Rust documentation and resources, including the official Rust website, The Book, standard library API reference, Rust by Example, and the Cargo Guide.

```Rust
Rust Resources:
  - Rust Website: https://www.rust-lang.org/
  - The Book: https://doc.rust-lang.org/book/
  - Standard Library API Reference: https://doc.rust-lang.org/std/
  - Rust by Example: https://doc.rust-lang.org/rust-by-example/
  - Cargo Guide: https://doc.rust-lang.org/cargo/guide/
  - Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/prelude/static

A collection of links to essential Rust documentation and resources, including the official Rust website, The Book, Standard Library API Reference, Rust by Example, the Cargo Guide, and Clippy documentation.

```rust
Rust Website: https://www.rust-lang.org/
The Book: https://doc.rust-lang.org/book/
Standard Library API Reference: https://doc.rust-lang.org/std/
Rust by Example: https://doc.rust-lang.org/rust-by-example/
The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/prelude/gather/index

Provides links to essential Rust documentation and resources, including The Book, Standard Library API Reference, Rust by Example, and the Cargo Guide. These are valuable for learning and using Rust effectively.

```Rust
rust_resources:
  - Rust website: https://www.rust-lang.org/
  - The Book: https://doc.rust-lang.org/book/
  - Standard Library API Reference: https://doc.rust-lang.org/std/
  - Rust by Example: https://doc.rust-lang.org/rust-by-example/
  - The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
  - Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/prelude/macro

Provides links to essential Rust resources, including the official Rust website, The Book, Standard Library API Reference, Rust by Example, and the Cargo Guide. These are valuable for learning and using Rust.

```Rust
rust_resources:
  - website: https://www.rust-lang.org/
  - book: https://doc.rust-lang.org/book/
  - std_api: https://doc.rust-lang.org/std/
  - by_example: https://doc.rust-lang.org/rust-by-example/
  - cargo_guide: https://doc.rust-lang.org/cargo/guide/
  - clippy: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/prelude/nan_propagating_aggregate/index

Offers links to essential Rust documentation and resources, including The Book, Standard Library API, Rust by Example, and Cargo Guide.

```Rust
Rust Website: https://www.rust-lang.org/
The Book: https://doc.rust-lang.org/book/
Standard Library API Reference: https://doc.rust-lang.org/std/
Rust by Example: https://doc.rust-lang.org/rust-by-example/
The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/prelude/iceberg/index

A collection of links to essential Rust documentation and resources, including the official Rust website, The Book, Standard Library API, Rust by Example, Cargo Guide, and Clippy.

```rust
Rust Resources:
  - Rust website: https://www.rust-lang.org/
  - The Book: https://doc.rust-lang.org/book/
  - Standard Library API Reference: https://doc.rust-lang.org/std/
  - Rust by Example: https://doc.rust-lang.org/rust-by-example/
  - The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
  - Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/error/constants/static

A collection of links to essential Rust resources, including the official Rust website, The Book, Standard Library API Reference, Rust by Example, Cargo Guide, and Clippy documentation.

```Rust
Rust Website: https://www.rust-lang.org/
The Book: https://doc.rust-lang.org/book/
Standard Library API Reference: https://doc.rust-lang.org/std/
Rust by Example: https://doc.rust-lang.org/rust-by-example/
The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Box Creation Example

Source: https://docs.rs/polars/latest/polars/series/type

Demonstrates the basic creation of a Box containing an integer.

```rust
let five = Box::new(5);
```

---

### Rust Box new_uninit Example

Source: https://docs.rs/polars/latest/polars/series/type

Shows how to create an uninitialized Box and defer its initialization, then safely assume its initialized state.

```rust
let mut five = Box::<u32>::new_uninit();
// Deferred initialization:
five.write(5);
let five = unsafe { five.assume_init() };

assert_eq!(*five, 5)
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/series/arithmetic/checked/index

Provides links to key resources within the Rust ecosystem, including the official Rust website, The Book, Standard Library API Reference, Rust by Example, and the Cargo Guide. These resources are valuable for learning and using Rust.

```Rust
Rust Website: https://www.rust-lang.org/
The Book: https://doc.rust-lang.org/book/
Standard Library API Reference: https://doc.rust-lang.org/std/
Rust by Example: https://doc.rust-lang.org/rust-by-example/
The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust String Trim Start Matches Examples

Source: https://docs.rs/polars/latest/polars/error/struct

Demonstrates the usage of `trim_start_matches` in Rust for removing characters or patterns from the beginning of a string. It shows examples with single characters, numeric characters, and character slices.

```rust
assert_eq!("11foo1bar11".trim_start_matches('1'), "foo1bar11");
assert_eq!("123foo1bar123".trim_start_matches(char::is_numeric), "foo1bar123");

let x: &[_] = &['1', '2'];
assert_eq!("12foo1bar12".trim_start_matches(x), "foo1bar12");
```

---

### Polars Quickstart: LazyFrame Operations

Source: https://docs.rs/polars/latest/polars/index

Demonstrates building queries with polars-lazy, including scanning Parquet files, grouping, aggregating with complex expressions, joining DataFrames, and materializing the result.

```rust
use polars::prelude::*;

let lf1 = LazyFrame::scan_parquet("myfile_1.parquet", Default::default())?
    .group_by([col("ham")])
    .agg([
        // expressions can be combined into powerful aggregations
        col("foo")
            .sort_by([col("ham").rank(Default::default(), None)], SortMultipleOptions::default())
            .last()
            .alias("last_foo_ranked_by_ham"),
        // every expression runs in parallel
        col("foo").cum_min(false).alias("cumulative_min_per_group"),
        // every expression runs in parallel
        col("foo").reverse().implode().alias("reverse_group"),
    ]);

let lf2 = LazyFrame::scan_parquet("myfile_2.parquet", Default::default())?
    .select([col("ham"), col("spam")]);

let df = lf1
    .join(lf2, [col("reverse")], [col("foo")], JoinArgs::new(JoinType::Left))
    // now we finally materialize the result.
    .collect()?;
```

---

### Rust Standard Library and Tools

Source: https://docs.rs/polars/latest/polars/datatypes/categorical/index

Provides links to essential Rust resources, including the official Rust website, The Book, Standard Library API Reference, Rust by Example, The Cargo Guide, and Clippy Documentation.

```Rust
Rust Resources:
  Website: https://www.rust-lang.org/
  The Book: https://doc.rust-lang.org/book/
  Standard Library API Reference: https://doc.rust-lang.org/std/
  Rust by Example: https://doc.rust-lang.org/rust-by-example/
  Cargo Guide: https://doc.rust-lang.org/cargo/guide/
  Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust String Trim Start Matches Examples

Source: https://docs.rs/polars/latest/polars/datatypes/time_zone/struct

Demonstrates the usage of `trim_start_matches` in Rust for removing characters or patterns from the beginning of a string. It shows examples with single characters, numeric characters, and character arrays.

```rust
assert_eq!("11foo1bar11".trim_start_matches('1'), "foo1bar11");
assert_eq!("123foo1bar123".trim_start_matches(char::is_numeric), "foo1bar123");

let x: &[_] = &['1', '2'];
assert_eq!("12foo1bar12".trim_start_matches(x), "foo1bar12");
```

---

### Rust Documentation Resources

Source: https://docs.rs/polars/latest/polars/prelude/datatypes/time_zone/index

A collection of links to essential Rust documentation resources, including the official Rust website, The Book, Standard Library API Reference, Rust by Example, The Cargo Guide, and Clippy Documentation.

```Rust
Rust Resources:
  - Rust website: https://www.rust-lang.org/
  - The Book: https://doc.rust-lang.org/book/
  - Standard Library API Reference: https://doc.rust-lang.org/std/
  - Rust by Example: https://doc.rust-lang.org/rust-by-example/
  - The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
  - Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/prelude/udf/index

Provides links to essential Rust programming resources, including the official Rust website, The Book, Standard Library API Reference, Rust by Example, and the Cargo Guide. These resources are fundamental for learning and using Rust.

```Rust
Rust Website: https://www.rust-lang.org/
The Book: https://doc.rust-lang.org/book/
Standard Library API Reference: https://doc.rust-lang.org/std/
Rust by Example: https://doc.rust-lang.org/rust-by-example/
Cargo Guide: https://doc.rust-lang.org/cargo/guide/
Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Get Glob Start Index

Source: https://docs.rs/polars/latest/polars/prelude/index

Gets the index of the first occurrence of a glob symbol in a string.

```rust
fn get_glob_start_idx(s: &str) -> Option<usize>
```

---

### Polars Prelude Modules Overview

Source: https://docs.rs/polars/latest/polars/prelude/index

This section provides a high-level overview of the modules available in the Polars prelude. Each entry links to the specific module documentation and indicates the associated Polars crate.

```rust
mod _csv_read_internal: polars-io
mod _internal: polars-io
mod aggregations
mod arity
mod array: polars-ops
mod binary: lazy
mod buffer: polars-io
mod byte_source: polars-io
mod cat: lazy
mod chunkedarray: temporal
  Traits and utilities for temporal data.
mod cloud: polars-io
  Interface with cloud storage through the object_store crate.
mod compression: polars-io
mod concat_arr: polars-ops
mod datatypes
  Data types supported by Polars.
mod datetime: polars-ops
mod default_arrays
mod deletion: lazy
mod dt: lazy
mod expr
mod file: polars-io
mod fill_null
mod fixed_size_list
mod float_sorted_arg_max
mod full
mod function_expr: lazy
mod gather
mod iceberg
  TODO
mod interpolate: polars-ops
mod interpolate_by: polars-ops
mod mkdir: polars-io
mod mode: polars-ops
mod named_serde: lazy
mod nan_propagating_aggregate: polars-ops
mod null
mod replace: temporal
mod round: polars-ops
mod row_encode
mod schema_inference: polars-io
mod search_sorted
```

---

### Rust Documentation Resources

Source: https://docs.rs/polars/latest/polars/chunked_array/index

Provides links to essential Rust documentation resources, including the official Rust website, The Book, Standard Library API Reference, Rust by Example, and The Cargo Guide.

```Rust
Rust Resources:
- Rust website: https://www.rust-lang.org/
- The Book: https://doc.rust-lang.org/book/
- Standard Library API Reference: https://doc.rust-lang.org/std/
- Rust by Example: https://doc.rust-lang.org/rust-by-example/
- The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
- Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Polars Directory and Serde Utilities

Source: https://docs.rs/polars/latest/polars/all

Documentation for functions related to creating directories recursively and setting up named Serde registries.

```rust
/// Creates a directory recursively.
fn mkdir_recursive(path: &str) -> PolarsResult<()>;

/// Creates a directory recursively using Tokio.
fn tokio_mkdir_recursive(path: &str) -> PolarsResult<()>;

/// Sets the named Serde registry.
fn set_named_serde_registry(registry: NamedSerdeRegistry) -> PolarsResult<()>;

```

---

### Rust Vec len() Example

Source: https://docs.rs/polars/latest/polars/frame/group_by/type

Illustrates the usage of the `len` method to get the number of elements currently in a vector.

```rust
let a = vec![1, 2, 3];
assert_eq!(a.len(), 3);
```

---

### Rust bytes() Example

Source: https://docs.rs/polars/latest/polars/prelude/datatypes/time_zone/struct

Illustrates how to use the `bytes()` method to get an iterator over the individual bytes of a string slice. This is useful for low-level byte processing.

```rust
let mut bytes = "bors".bytes();

assert_eq!(Some(b'b'), bytes.next());
assert_eq!(Some(b'o'), bytes.next());
assert_eq!(Some(b'r'), bytes.next());
assert_eq!(Some(b's'), bytes.next());

assert_eq!(None, bytes.next());
```

---

### Polars Prelude Modules

Source: https://docs.rs/polars/latest/polars/prelude/type

Lists the available modules within the Polars prelude, offering a categorized view of the library's components.

```rust
mod _csv_read_internal;
mod _internal;
mod aggregations;
mod arity;
mod array;
mod binary;
mod buffer;
mod byte_source;
mod cat;
mod chunkedarray;
mod cloud;
mod compression;
mod concat_arr;
mod datatypes;
mod datetime;
mod default_arrays;
mod deletion;
mod dt;
mod expr;
mod file;
mod fill_null;
mod fixed_size_list;
mod float_sorted_arg_max;
mod full;
mod function_expr;
mod gather;
mod iceberg;
mod interpolate;
mod interpolate_by;
mod mkdir;
mod mode;
mod named_serde;
mod nan_propagating_aggregate;
mod null;
mod replace;
mod round;
mod row_encode;
mod schema_inference;
mod search_sorted;
mod series;
mod sort;
mod strings;
mod sync_on_close;
mod udf;
mod utf8;
mod zip;
```

---

### Rust Vec into_iter Example

Source: https://docs.rs/polars/latest/polars/frame/group_by/type

Demonstrates creating a consuming iterator from a Vec<T>. The vector cannot be used after calling this method. It iterates from start to end, moving each value out.

```rust
let v = vec!["a".to_string(), "b".to_string()];
let mut v_iter = v.into_iter();

let first_element: Option<String> = v_iter.next();

assert_eq!(first_element, Some("a".to_string()));
assert_eq!(v_iter.next(), Some("b".to_string()));
assert_eq!(v_iter.next(), None);
```

---

### Polars Modules Overview

Source: https://docs.rs/polars/latest/polars/prelude/cloud/index

Lists the various modules available in the Polars prelude, each offering specific functionalities for data processing and analysis.

```rust
// Modules for data manipulation and analysis:
// use polars::prelude::aggregations;
// use polars::prelude::chunkedarray;
// use polars::prelude::datatypes;
// use polars::prelude::expr;
// use polars::prelude::file;
// use polars::prelude::series;
// use polars::prelude::strings;
// use polars::prelude::utf8;

// Modules for specific data types and operations:
// use polars::prelude::array;
// use polars::prelude::binary;
// use polars::prelude::cat;
// use polars::prelude::datetime;
// use polars::prelude::fixed_size_list;
// use polars::prelude::float_sorted_arg_max;
// use polars::prelude::full;
// use polars::prelude::gather;
// use polars::prelude::interpolate;
// use polars::prelude::interpolate_by;
// use polars::prelude::mode;
// use polars::prelude::null;
// use polars::prelude::replace;
// use polars::prelude::round;
// use polars::prelude::sort;
// use polars::prelude::udf;
// use polars::prelude::zip;

// Modules for internal or specialized functionalities:
// use polars::prelude::_csv_read_internal;
// use polars::prelude::_internal;
// use polars::prelude::arity;
// use polars::prelude::buffer;
// use polars::prelude::byte_source;
// use polars::prelude::cloud;
// use polars::prelude::compression;
// use polars::prelude::concat_arr;
// use polars::prelude::default_arrays;
// use polars::prelude::deletion;
// use polars::prelude::dt;
// use polars::prelude::fill_null;
// use polars::prelude::function_expr;
// use polars::prelude::iceberg;
// use polars::prelude::mkdir;
// use polars::prelude::named_serde;
// use polars::prelude::nan_propagating_aggregate;
// use polars::prelude::row_encode;
// use polars::prelude::schema_inference;
// use polars::prelude::search_sorted;
// use polars::prelude::sync_on_close;

```

---

### Rust Box try_new Example

Source: https://docs.rs/polars/latest/polars/series/type

Shows how to attempt to allocate memory and create a Box, returning an error if allocation fails. Requires the 'allocator_api' feature.

```rust
#![feature(allocator_api)]

let five = Box::try_new(5)?;
```

---

### Get Ordinal Day of Year (Polars)

Source: https://docs.rs/polars/latest/polars/prelude/datatypes/categorical/type

The `ordinal` method returns the day of the year for each element in a datetime series, starting from 1. It operates on ChunkedArrays of Int16Type.

```rust
fn ordinal(&self) -> ChunkedArray<Int16Type>
```

---

### SortOptions Documentation

Source: https://docs.rs/polars/latest/polars/prelude/sort/options/struct

Documentation for the SortOptions struct in the Polars library, including a link to an example.

```Rust
struct SortOptions {
    // Fields related to sorting options
}

// Example usage of SortOptions
```

---

### Polars Library Overview

Source: https://docs.rs/polars/latest/polars/prelude/search_sorted/fn

Provides an overview of the Polars library, version 0.50.0. Includes links to the library's documentation, source code, and related resources.

```Rust
Project: /websites/rs-polars-polars
Content:
[ Docs.rs ](https://docs.rs/)
  * [ polars-0.50.0 ](https://docs.rs/polars/latest/polars/prelude/search_sorted/fn.binary_search_ca.html "DataFrame library based on Apache Arrow")
    * polars 0.50.0
    * [](https://docs.rs/polars/0.50.0/polars/prelude/search_sorted/fn.binary_search_ca.html "Get a link to this specific version")
    * [ ](https://docs.rs/crate/polars/latest "See polars in docs.rs")
    * [MIT](https://spdx.org/licenses/MIT)
    * Links
    * [ ](https://www.pola.rs/)
    * [ ](https://github.com/pola-rs/polars)
    * [ ](https://crates.io/crates/polars "See polars in crates.io")
    * [ ](https://docs.rs/crate/polars/latest/source/ "Browse source of polars-0.50.0")
    * Owners
    * [ ](https://crates.io/users/ritchie46)
    * [ ](https://crates.io/users/stijnherfst)
    * Dependencies
    *       * [ polars-arrow ^0.50.0 _normal_ ](https://docs.rs/polars-arrow/^0.50.0)
      * [ polars-core ^0.50.0 _normal_ ](https://docs.rs/polars-core/^0.50.0)
      * [ polars-error ^0.50.0 _normal_ ](https://docs.rs/polars-error/^0.50.0)
      * [ polars-io ^0.50.0 _normal_ _optional_ ](https://docs.rs/polars-io/^0.50.0)
      * [ polars-lazy ^0.50.0 _normal_ _optional_ ](https://docs.rs/polars-lazy/^0.50.0)
      * [ polars-ops ^0.50.0 _normal_ _optional_ ](https://docs.rs/polars-ops/^0.50.0)
      * [ polars-parquet ^0.50.0 _normal_ ](https://docs.rs/polars-parquet/^0.50.0)
      * [ polars-plan ^0.50.0 _normal_ _optional_ ](https://docs.rs/polars-plan/^0.50.0)
      * [ polars-sql ^0.50.0 _normal_ _optional_ ](https://docs.rs/polars-sql/^0.50.0)
      * [ polars-time ^0.50.0 _normal_ _optional_ ](https://docs.rs/polars-time/^0.50.0)
      * [ polars-utils ^0.50.0 _normal_ ](https://docs.rs/polars-utils/^0.50.0)
      * [ apache-avro ^0.17 _dev_ ](https://docs.rs/apache-avro/^0.17)
      * [ polars-arrow ^0.50.0 _dev_ ](https://docs.rs/polars-arrow/^0.50.0)
      * [ avro-schema ^0.3 _dev_ ](https://docs.rs/avro-schema/^0.3)
      * [ chrono ^0.4.31 _dev_ ](https://docs.rs/chrono/^0.4.31)
      * [ either ^1.14 _dev_ ](https://docs.rs/either/^1.14)
      * [ ethnum ^1 _dev_ ](https://docs.rs/ethnum/^1)
      * [ futures ^0.3.25 _dev_ ](https://docs.rs/futures/^0.3.25)
      * [ proptest ^1.6 _dev_ ](https://docs.rs/proptest/^1.6)
      * [ rand ^0.9 _dev_ ](https://docs.rs/rand/^0.9)
      * [ tokio ^1.44 _dev_ ](https://docs.rs/tokio/^1.44)
      * [ tokio-util ^0.7.8 _dev_ ](https://docs.rs/tokio-util/^0.7.8)
      * [ version_check ^0.9.4 _build_ ](https://docs.rs/version_check/^0.9.4)
      * [ getrandom ^0.3 _normal_ ](https://docs.rs/getrandom/^0.3)
      * [ getrandom ^0.2 _normal_ ](https://docs.rs/getrandom/^0.2)
    * Versions
    * [ **66.67%** of the crate is documented ](https://docs.rs/crate/polars/latest)
  * [ Platform ](https://docs.rs/polars/latest/polars/prelude/search_sorted/fn.binary_search_ca.html)
    * [i686-unknown-linux-gnu](https://docs.rs/crate/polars/latest/target-redirect/i686-unknown-linux-gnu/polars/prelude/search_sorted/fn.binary_search_ca.html)
    * [x86_64-unknown-linux-gnu](https://docs.rs/crate/polars/latest/target-redirect/x86_64-unknown-linux-gnu/polars/prelude/search_sorted/fn.binary_search_ca.html)
  * [ Feature flags ](https://docs.rs/crate/polars/latest/features "Browse available feature flags of polars-0.50.0")


  * [docs.rs](https://docs.rs/polars/latest/polars/prelude/search_sorted/fn.binary_search_ca.html)
    * [](https://docs.rs/about)
    * [](https://docs.rs/about/badges)
    * [](https://docs.rs/about/builds)
    * [](https://docs.rs/about/metadata)
    * [](https://docs.rs/about/redirections)
    * [](https://docs.rs/about/download)
    * [](https://docs.rs/about/rustdoc-json)
    * [](https://docs.rs/releases/queue)
    * [](https://foundation.rust-lang.org/policies/privacy-policy/#docs.rs)


  * [Rust](https://docs.rs/polars/latest/polars/prelude/search_sorted/fn.binary_search_ca.html)
    * [Rust website](https://www.rust-lang.org/)
    * [The Book](https://doc.rust-lang.org/book/)
    * [Standard Library API Reference](https://doc.rust-lang.org/std/)
    * [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
    * [The Cargo Guide](https://doc.rust-lang.org/cargo/guide/)
    * [Clippy Documentation](https://doc.rust-lang.org/nightly/clippy)
```

---

### Get Range Slice (Rust)

Source: https://docs.rs/polars/latest/polars/prelude/iceberg/struct

Returns an immutable slice of key-value pairs within a specified range of indices. The range can be defined using standard Rust range syntax (e.g., `start..end`, `start..=end`). Valid indices must be within the schema's bounds. This operation has a time complexity of O(1).

```rust
pub fn get_range<R>(&self, range: R) -> Option<&Slice<K, V>>
where R: RangeBounds<usize>,
// Returns a slice of key-value pairs in the given range of indices.
// Valid indices are `0 <= index < self.len()`.
// Computes in **O(1)** time.
```

---

### Manually Create Box from Scratch with System Allocator

Source: https://docs.rs/polars/latest/polars/series/type

Shows how to manually allocate memory using the system allocator, write data to it, and then construct a Box from the raw pointer. Requires `allocator_api` and `slice_ptr_get` features.

```rust
#![feature(allocator_api, slice_ptr_get)]

use std::alloc::{Allocator, Layout, System};

unsafe {
    let ptr = System.allocate(Layout::new::<i32>())?.as_mut_ptr() as *mut i32;
    // In general .write is required to avoid attempting to destruct
    // the (uninitialized) previous contents of `ptr`, though for this
    // simple example `*ptr = 5` would have worked as well.
    ptr.write(5);
    let x = Box::from_raw_in(ptr, System);
}
```

---

### Rust Arc Weak Count Example

Source: https://docs.rs/polars/latest/polars/prelude/iceberg/type

Demonstrates how to get the weak reference count of an Arc pointer. This is useful for understanding how many weak references exist to a shared resource.

```rust
use std::sync::Arc;

let five = Arc::new(5);
let _weak_five = Arc::downgrade(&five);

// This assertion is deterministic because we haven't shared
// the `Arc` or `Weak` between threads.
assert_eq!(1, Arc::weak_count(&five));
```

---

### Get DataFrame Columns (Rust)

Source: https://docs.rs/polars/latest/polars/frame/struct

Provides an example of retrieving an immutable slice of `Column` objects from a Polars DataFrame. This is useful for inspecting the structure and names of the DataFrame's columns.

```rust
let df: DataFrame = df!("Name" => ["Adenine", "Cytosine", "Guanine", "Thymine"],
                        "Symbol" => ["A", "C", "G", "T"])?;
let columns: &[Column] = df.get_columns();

assert_eq!(columns[0].name(), "Name");
assert_eq!(columns[1].name(), "Symbol");
```

---

### Rust rmatch_indices Example

Source: https://docs.rs/polars/latest/polars/datatypes/time_zone/struct

Demonstrates the usage of `rmatch_indices` for finding all occurrences of a substring in reverse order within a string. It returns tuples of the starting byte index and the matched substring.

```rust
let v: Vec<_> = "abcXXXabcYYYabc".rmatch_indices("abc").collect();
assert_eq!(v, [(12, "abc"), (6, "abc"), (0, "abc")]);

let v: Vec<_> = "1abcabc2".rmatch_indices("abc").collect();
assert_eq!(v, [(4, "abc"), (1, "abc")]);

let v: Vec<_> = "ababa".rmatch_indices("aba").collect();
assert_eq!(v, [(2, "aba")]); // only the last `aba`
```

---

### Polars Buffer Module Overview

Source: https://docs.rs/polars/latest/polars/prelude/buffer/struct

Provides an overview of the polars::prelude::buffer module, listing available structs, enums, and functions.

```rust
Structs:
  CategoricalField
  DatetimeField
  Utf8Field

Enums:
  Buffer

Functions:
  init_buffers
  validate_utf8
```

---

### Rust Ecosystem Links

Source: https://docs.rs/polars/latest/polars/prelude/chunkedarray/string/enum

This snippet provides links to essential Rust documentation and resources, including the official Rust website, The Book, Standard Library API Reference, Rust by Example, the Cargo Guide, and Clippy Documentation.

```Rust
Rust Website: https://www.rust-lang.org/
The Book: https://doc.rust-lang.org/book/
Standard Library API Reference: https://doc.rust-lang.org/std/
Rust by Example: https://doc.rust-lang.org/rust-by-example/
The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Polars Prelude Miscellaneous Functions

Source: https://docs.rs/polars/latest/polars/prelude/mkdir/index

Contains various other utility functions for Polars, including creating integer ranges, handling schemas, and preparing data for cloud operations.

```rust
/// Creates a new Series with an integer range.
fn new_int_range(start: i32, end: i32) -> Series;

/// Overwrites the schema of a DataFrame.
fn overwrite_schema(df: &DataFrame, schema: Schema) -> DataFrame;

/// Resolves a path to a user's home directory.
fn resolve_homedir(path: &str) -> String;

/// Computes the run-length encoding of a Series.
fn rle(s: &Series) -> Series;

/// Computes the run-length encoding IDs of a Series.
fn rle_id(s: &Series) -> Series;

/// Computes the run-length encoding lengths of a Series.
fn rle_lengths(s: &Series) -> Series;

/// Reduces a Series using a given expression.
fn reduce_exprs(s: &Series, expr: Expr) -> Series;

/// Removes the Byte Order Mark (BOM) from strings in a Series.
fn remove_bom(s: &Series) -> Series;

/// Repeats values in a Series a specified number of times.
fn repeat(s: &Series, n: usize) -> Series;

/// Replaces date components in a Series.
fn replace_date(s: &Series, date: NaiveDate) -> Series;

/// Replaces datetime components in a Series.
fn replace_datetime(s: &Series, dt: NaiveDateTime) -> Series;

/// Replaces time zone information in a Series.
fn replace_time_zone(s: &Series, tz: &str) -> Series;

/// Computes the nth element of a Series.
fn nth(s: &Series, n: usize) -> Option<AnyValue>;

/// Creates a DataFrame with a materialized projection.
fn materialize_projection(df: &DataFrame, projection: Vec<usize>) -> DataFrame;

/// Creates an empty DataFrame with a materialized schema.
fn materialize_empty_df(schema: Schema) -> DataFrame;

/// Merges data types of Series.
fn merge_dtypes(s1: &Series, s2: &Series) -> PolarsResult<DataType>;

/// Prepares a cloud plan.
fn prepare_cloud_plan(plan: CloudPlan) -> CloudPlan;

/// Performs a private left join with multiple keys.
fn private_left_join_multiple_keys(df1: &DataFrame, df2: &DataFrame, left_on: Vec<String>, right_on: Vec<String>) -> PolarsResult<DataFrame>;

/// Computes the quantile of a Series.
fn quantile(s: &Series, quantile: f64) -> Option<f64>;

/// Creates a linear space for f32 values.
fn new_linear_space_f32(start: f32, end: f32, n: usize) -> Series;

/// Creates a linear space for f64 values.
fn new_linear_space_f64(start: f64, end: f64, n: usize) -> Series;

/// Creates a new integer range Series.
fn new_int_range(start: i64, end: i64) -> Series;

/// Gets the last element of a Series.
fn last(s: &Series) -> Option<AnyValue>;

/// Counts leading ones in a Series.
fn leading_ones(s: &Series) -> Series;

/// Counts leading zeros in a Series.
fn leading_zeros(s: &Series) -> Series;

/// Gets the length of a Series.
fn len(s: &Series) -> usize;
```

---

### Rust Check Character Boundary

Source: https://docs.rs/polars/latest/polars/prelude/datatypes/time_zone/struct

Illustrates the `is_char_boundary` method, which verifies if a given byte index in a string corresponds to the start or end of a UTF-8 character sequence. It includes examples with multi-byte characters.

```rust
let s = "Löwe 老虎 Léopard";
assert!(s.is_char_boundary(0));
// start of `老`
assert!(s.is_char_boundary(6));
assert!(s.is_char_boundary(s.len()));

// second byte of `ö`
assert!(!s.is_char_boundary(2));

// third byte of `老`
assert!(!s.is_char_boundary(8));
```

---

### Column Write Options and Glob Start Index

Source: https://docs.rs/polars/latest/polars/prelude/mkdir/index

Retrieves column write options and the starting index for glob patterns. Useful for configuring data output and parsing file paths.

```rust
fn get_column_write_options(df: &DataFrame) -> Vec<ColumnWriteOptions>;
fn get_glob_start_idx(path: &str) -> usize;
```

---

### Rust: Constructing Pin<Box<T, A>> with System Allocator

Source: https://docs.rs/polars/latest/polars/series/type

Demonstrates the creation of a pinned Box using the System allocator and initializing it with zeroed memory. This is an experimental nightly-only feature.

```rust
#![feature(allocator_api)]

use std::alloc::System;

let zero = Box::<u32, _>::try_new_zeroed_in(System)?;
let zero = unsafe { zero.assume_init() };

assert_eq!(*zero, 0);
```

---

### Column Write Options and Glob Start Index

Source: https://docs.rs/polars/latest/polars/prelude/binary/index

Retrieves column write options and the starting index for glob patterns. Useful for configuring data output and parsing file paths.

```rust
fn get_column_write_options(df: &DataFrame) -> Vec<ColumnWriteOptions>;
fn get_glob_start_idx(path: &str) -> usize;
```

---

### Rust Box Examples

Source: https://docs.rs/polars/latest/polars/series/amortized_iter/type

Demonstrates the creation and comparison of boxed values in Rust.

```rust
let x = 5;
let boxed = Box::new(5);

assert_eq!(Box::from(x), boxed);
```

---

### Polars Prelude Modules

Source: https://docs.rs/polars/latest/polars/prelude/compression/index

Lists the available modules within the Polars prelude, covering various functionalities like CSV reading, aggregations, data types, and more.

```rust
/// Modules in Polars Prelude:
/// - _csv_read_internal
/// - _internal
/// - aggregations
/// - arity
/// - array
/// - binary
/// - buffer
/// - byte_source
/// - cat
/// - chunkedarray
/// - cloud
/// - compression
/// - concat_arr
/// - datatypes
/// - datetime
/// - default_arrays
/// - deletion
/// - dt
/// - expr
/// - file
/// - fill_null
/// - fixed_size_list
/// - float_sorted_arg_max
/// - full
/// - function_expr
/// - gather
/// - iceberg
/// - interpolate
/// - interpolate_by
/// - mkdir
/// - mode
/// - named_serde
/// - nan_propagating_aggregate
/// - null
/// - replace
/// - round
/// - row_encode
/// - schema_inference
/// - search_sorted
/// - series
/// - sort
/// - strings
/// - sync_on_close
/// - udf
/// - utf8
/// - zip
```

---

### Column Write Options and Glob Start Index

Source: https://docs.rs/polars/latest/polars/prelude/sync_on_close/index

Retrieves column write options and the starting index for glob patterns. Useful for configuring data output and parsing file paths.

```rust
fn get_column_write_options(df: &DataFrame) -> Vec<ColumnWriteOptions>;
fn get_glob_start_idx(path: &str) -> usize;
```

---

### Column Write Options and Glob Start Index

Source: https://docs.rs/polars/latest/polars/prelude/zip/index

Retrieves column write options and the starting index for glob patterns. Useful for configuring data output and parsing file paths.

```rust
fn get_column_write_options(df: &DataFrame) -> Vec<ColumnWriteOptions>;
fn get_glob_start_idx(path: &str) -> usize;
```

---

### Column Write Options and Glob Start Index

Source: https://docs.rs/polars/latest/polars/prelude/static

Retrieves column write options and the starting index for glob patterns. Useful for configuring data output and parsing file paths.

```rust
fn get_column_write_options(df: &DataFrame) -> Vec<ColumnWriteOptions>;
fn get_glob_start_idx(path: &str) -> usize;
```

---

### Polars Partitioning and Plan Serialization

Source: https://docs.rs/polars/latest/polars/all

Documentation for types related to data partitioning and plan serialization, including PartitionSinkType, PartitionSinkTypeIR, PartitionTargetContext, PartitionTargetContextKey, and PlanSerializationContext.

```rust
/// Type of partition sink.
enum PartitionSinkType;

/// Intermediate representation of partition sink type.
enum PartitionSinkTypeIR;

/// Context for partition targets.
struct PartitionTargetContext;

/// Key for partition target context.
struct PartitionTargetContextKey;

/// Context for serializing query plans.
struct PlanSerializationContext;
```

---

### Rust Standard Library and Ecosystem

Source: https://docs.rs/polars/latest/polars/prelude/null/index

Provides links to key Rust documentation resources, including the official website, The Book, standard library API reference, and Rust by Example.

```Rust
Rust Resources:
  Rust website: https://www.rust-lang.org/
  The Book: https://doc.rust-lang.org/book/
  Standard Library API Reference: https://doc.rust-lang.org/std/
  Rust by Example: https://doc.rust-lang.org/rust-by-example/
  The Cargo Guide: https://doc.rust-lang.org/cargo/guide/
  Clippy Documentation: https://doc.rust-lang.org/nightly/clippy
```

---

### Rust match_indices Example

Source: https://docs.rs/polars/latest/polars/error/struct

Illustrates the `match_indices` method in Rust, which returns an iterator over the disjoint matches of a pattern within a string slice, along with their starting indices. Overlapping matches are handled by returning only the first occurrence.

```rust
let v: Vec<_> = "abcXXXabcYYYabc".match_indices("abc").collect();
assert_eq!(v, [(0, "abc"), (6, "abc"), (12, "abc")]);

let v: Vec<_> = "1abcabc2".match_indices("abc").collect();
assert_eq!(v, [(1, "abc"), (4, "abc")]);

let v: Vec<_> = "ababa".match_indices("aba").collect();
assert_eq!(v, [(0, "aba")]); // only the first `aba`
```

---

### Column Write Options and Glob Start Index

Source: https://docs.rs/polars/latest/polars/prelude/strings/index

Retrieves column write options and the starting index for glob patterns. Useful for configuring data output and parsing file paths.

```rust
fn get_column_write_options(df: &DataFrame) -> Vec<ColumnWriteOptions>;
fn get_glob_start_idx(path: &str) -> usize;
```

---

### Polars-Utils Key Trait Methods

Source: https://docs.rs/polars/latest/polars/series/arithmetic/struct

Documentation for the `Key` trait's core methods: `init`, `get`, and `drop_in_place`. These methods are used for managing data in memory, including initialization, safe retrieval of references, and proper cleanup.

```APIDOC
Key Trait:
  init(ptr: [*mut u8])
    Initializes the key in the given memory location.
    Parameters:
      ptr: A mutable pointer to a byte array where the key will be initialized.

  get(ptr: [*const u8]) -> &'a T
    Gets a reference to the key from the given memory location.
    Parameters:
      ptr: A constant pointer to a byte array from which to retrieve the key.
    Returns: A reference to the key of type T.

  drop_in_place(ptr: [*mut u8])
    Drops the key in place, freeing associated resources.
    Parameters:
      ptr: A mutable pointer to a byte array containing the key to be dropped.
```

---

### Polars Supported Platforms

Source: https://docs.rs/polars/latest/polars/error/index

Lists the target platforms for which Polars has been built and documented.

```Rust
Platforms:
  i686-unknown-linux-gnu
  x86_64-unknown-linux-gnu
```
