/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */
package jcompute.core.util.function;

import java.util.Objects;

@FunctionalInterface
public interface ByteUnaryOperator {

    /**
     * Applies this operator to the given operand.
     *
     * @param operand the operand
     * @return the operator result
     */
    byte applyAsByte(byte operand);

    /**
     * Returns a composed operator that first applies the {@code before}
     * operator to its input, and then applies this operator to the result.
     * If evaluation of either operator throws an exception, it is relayed to
     * the caller of the composed operator.
     *
     * @param before the operator to apply before this operator is applied
     * @return a composed operator that first applies the {@code before}
     * operator and then applies this operator
     * @throws NullPointerException if before is null
     *
     * @see #andThen(ByteUnaryOperator)
     */
    default ByteUnaryOperator compose(final ByteUnaryOperator before) {
        Objects.requireNonNull(before);
        return (final byte v) -> applyAsByte(before.applyAsByte(v));
    }

    /**
     * Returns a composed operator that first applies this operator to
     * its input, and then applies the {@code after} operator to the result.
     * If evaluation of either operator throws an exception, it is relayed to
     * the caller of the composed operator.
     *
     * @param after the operator to apply after this operator is applied
     * @return a composed operator that first applies this operator and then
     * applies the {@code after} operator
     * @throws NullPointerException if after is null
     *
     * @see #compose(ByteUnaryOperator)
     */
    default ByteUnaryOperator andThen(final ByteUnaryOperator after) {
        Objects.requireNonNull(after);
        return (final byte t) -> after.applyAsByte(applyAsByte(t));
    }

    /**
     * Returns a unary operator that always returns its input argument.
     *
     * @return a unary operator that always returns its input argument
     */
    static ByteUnaryOperator identity() {
        return t -> t;
    }

}