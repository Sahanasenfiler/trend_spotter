# check_import.py

print("--- Starting Diagnostic Test ---")

try:
    # First, let's see if we can import the file at all.
    import reddit_fetcher
    print("✅ SUCCESS: The module 'reddit_fetcher.py' was imported successfully.")

    # Now, let's see what names are actually inside it.
    print("\nInspecting the contents of the 'reddit_fetcher' module...")
    
    # hasattr() is the definitive way to check if a name exists.
    if hasattr(reddit_fetcher, 'RedditFetcher'):
        print("✅ SUCCESS: The class 'RedditFetcher' was found inside the module.")
        print("\nDIAGNOSIS: The problem might be intermittent. Please try running the main script again.")
    else:
        print("❌ FAILURE: The class 'RedditFetcher' was NOT found inside the module.")
        print("\nDIAGNOSIS: This confirms the problem is with the content of 'reddit_fetcher.py'.")
        print("The most likely cause is that the file was not saved after pasting the correct code, or there is a typo in the class name.")
    
    # Let's print all available names to see what's really there.
    print("\n--- All available names in reddit_fetcher: ---")
    # This will show us everything, like 'praw', 'pd', and hopefully 'RedditFetcher'
    print(dir(reddit_fetcher))
    print("--- End of name list ---")


except ImportError as e:
    print(f"❌ FAILURE: A critical ImportError occurred: {e}")
    print("\nDIAGNOSIS: Python cannot even find or parse 'reddit_fetcher.py'. This could be a circular import or a major syntax error in that file.")

except Exception as e:
    print(f"❌ FAILURE: A different error occurred during the import process: {e}")
    print("\nDIAGNOSIS: This points to a syntax error inside 'reddit_fetcher.py'.")

print("\n--- Diagnostic Test Finished ---")