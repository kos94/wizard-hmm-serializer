package com.wizardfight;

public enum Shape { 
	TRIANGLE("triangle"),
	CIRCLE("circle"), 
	CLOCK("clock"),
	Z("z"),
	V("v"),
	PI("pi"),
	SHIELD("shield"), 
	FAIL("fail"),
	NONE("none");
	
	private final String name;
	
	private Shape(String s) {
		name = s;
	}
	
	@Override
	public String toString() {
		return name;
	}
    }
