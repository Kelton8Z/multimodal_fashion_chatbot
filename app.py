import os
from pathlib import Path
from argparse import Namespace

import json

from jina import flow
from jina.helper import countdown

if __name__ == "__main__":
    from helper import (
        print_result,
        write_html,
        download_data,
        index_generator,
        query_generator,
        colored,
    )
    from executors import MyEncoder, MyIndexer, MyEvaluator
else:
    from .helper import (
        print_result,
        write_html,
        download_data,
        index_generator,
        query_generator,
        colored,
    )
    from .executors import MyEncoder, MyIndexer, MyEvaluator

cur_dir = os.path.dirname(os.path.abspath(__file__))


def hello_world(args):
    """
    Runs Jina's Hello World.

    Usage:
        Use it via CLI :command:`jina hello-world`.

    Description:
        It downloads Amazon fashion dataset and :term:`Indexer<indexes>` 50,000 images.
        The index is stored into 4 *shards*. It randomly samples 128 unseen images as :term:`Queries<Searching>`
        Results are shown in a webpage.


    :param args: Argparse object
    """

    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    targets = {
        "index-labels": {
            "label": args.items['product_name'],
            "filename": os.path.join(args.workdir, "index-labels"),
        },
        "index": {
            "image_urls": args.items['large'],
            "filename": os.path.join(args.workdir, "index-original"),
        },
    }

    # download the data
    download_data(targets, args.download_proxy)

    # reduce the network load by using `fp16`, or even `uint8`
    os.environ["JINA_ARRAY_QUANT"] = "fp16"
    os.environ["HW_WORKDIR"] = args.workdir

    # now comes the real work
    f = flow().add(uses=MyEncoder, parallel=2).add(uses=MyIndexer).add(uses=MyEvaluator)
    # run it!
    with f:
        f.index(
            index_generator(num_docs=targets["index"]["data"].shape[0], target=targets),
            request_size=args.request_size,
        )
        f.use_rest_gateway()
        # wait for couple of seconds
        countdown(
            3,
            reason=colored(
                "behold! im going to switch to query mode",
                "cyan",
                attrs=["underline", "bold", "reverse"],
            ),
        )

        """
        f.post(
            '/eval',
            query_generator(
                num_docs=args.num_query, target=targets, with_groundtruth=True
            ),
            shuffle=True,
            on_done=print_result,
            request_size=args.request_size,
            parameters={'top_k': args.top_k},
        )"""

        # write result to html
        write_html(os.path.join("static/chatbot.html"))
        """
        url_html_path = 'file://' + os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'static/chatbot.html'
            )
        )
        
        try:
            webbrowser.open(url_html_path, new=2)
        except:
            pass  # intentional pass, browser support isn't cross-platform
        finally:
            default_logger.success(
                f'You should see a demo page opened in your browser, '
                f'if not, you may open {url_html_path} manually'
            )"""

        f.block()

        f.search(
            query_generator(
                num_docs=args.num_query, target=targets, with_groundtruth=True
            ),
            shuffle=True,
            on_done=print_result,
            request_size=args.request_size,
            parameters={"top_k": args.top_k},
        )


if __name__ == "__main__":
    with open(
        "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson",
        "r",
    ) as f:
        items = f.read().split('\n')
        selected_fields = ["uniq_id", "product_url",
                            "product_name",
                            "large",
                            "brand",
                            "sales_price",
                            "rating",
                            "sales_rank_in_parent_category",
                            "sales_rank_in_child_category",
                            "delivery_type",
                            "meta_keywords",
                            "best_seller_tag__y_or_n",
                            "other_items_customers_buy",
                            "product_details__k_v_pairs"]
        filtered_items = []
        for item in items:
            try:
                item = json.loads(item)
                filtered_item = {}
                for k,v in item.items():
                    if k in selected_fields:
                        if k in ['large', 'other_items_customers_buy']:
                            filtered_item[k] = v.split('|')
                        filtered_item[k] = v
                filtered_items.append(filtered_item)
            except json.decoder.JSONDecodeError:
                print(item)
        # large_image_urls = items[0]['large']

        #for filtered_item in filtered_items:
        args = Namespace(workdir='292303e8-0083-43ed-b0c7-93cec48a1e88', download_proxy=None, items=filtered_items)
                         #index_data_urls='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                         #index_labels_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
#                  request_size=1024, num_query=128, top_k=50)
        hello_world(args)

"""
"root":{22 items
"uniq_id":string"26d41bdc1495de290bc8e6062d927729"
"crawl_timestamp":string"2020-02-07 05:11:36 +0000"
"asin":string"B07STS2W9T"
"product_url":string"https://www.amazon.in/Facon-Kalamkari-Handblock-Dancers-Lehenga/dp/B07SWVSRPP/"
"product_name":string"LA' Facon Cotton Kalamkari Handblock Saree Blouse Fabric 100 cms Black Base Dancers (Cotton)"
"image_urls__small":string"https://images-na.ssl-images-amazon.com/images/I/51Wj2WownyL._SR38,50_.jpg|https://images-na.ssl-images-amazon.com/images/I/51tylkSxAIL._SR38,50_.jpg|https://images-na.ssl-images-amazon.com/images/I/51345gshQ9L._SR38,50_.jpg"
"medium":string"https://images-na.ssl-images-amazon.com/images/I/51Wj2WownyL.jpg|https://images-na.ssl-images-amazon.com/images/I/51tylkSxAIL.jpg|https://images-na.ssl-images-amazon.com/images/I/51345gshQ9L.jpg"
"large":string"https://images-na.ssl-images-amazon.com/images/I/81MqmouZ9kL._UL1500_.jpg|https://images-na.ssl-images-amazon.com/images/I/814Tnuvt5kL._UL1500_.jpg|https://images-na.ssl-images-amazon.com/images/I/81fFr%2B%2Bd6TL._UL1500_.jpg"
"browsenode":string"1968255031"
"brand":string"LA' Facon"
"sales_price":string"200.00"
"weight":string"999999999"
"rating":string"5.0"
"sales_rank_in_parent_category":{1 item
"ClothingAccessories":string"#19,259"
}
"sales_rank_in_child_category":{1 item
"WomensKurtasKurtis":string"#1793"
}
"delivery_type":string"fulfilled_by_merchant"
"meta_keywords":string"LA' Facon Cotton Kalamkari Handblock Saree Blouse Fabric 100 cms Black Base Dancers (Cotton)"
"amazon_prime__y_or_n":string"N"
"parent___child_category__all":{2 items
"ClothingAccessories":string"#19,259"
"WomensKurtasKurtis":string"#1793"
}
"best_seller_tag__y_or_n":string"N"
"other_items_customers_buy":string"Cotton Kalamkari Handblock Saree Blouse/Kurti Fabric 100 cms - Base Dancer Print - Red Colour | RJFabrics Women's Printed Cotton Fabric (2.5 Meter Length 43 Inch Width) | Cotton Kalamkari Handblock Saree Blouse/Kurti Fabric 100 cms - Multi Base Dancer Print - Maroon Colour | Cotton Kalamkari Handblock Saree Blouse/Kurti Fabric 100 cms Black Colour - Multi Base Dancers Print | Cotton Kalamkari Handblock Saree Blouse/Kurti Fabric 100 cms - Hand Print - Cream Colour | Cotton Kalamkari Handblock Saree Blouse/Kurti Fabric 100 cms Red- Budda Print | Cotton Kalamkari Handblock Saree Blouse/Kurti Fabric 100 cms Black Colour - Multi Base Dancers Print | RJFabrics Women's Printed Cotton Fabric (2.5 Meter Length 43 Inch Width) | SanDisk 128GB Class 10 microSDXC Memory Card with Adapter (SDSQUAR-128G-GN6MA)"
"product_details__k_v_pairs":{5 items
"Item_part_number":string"Devi face"
"ASIN":string"B07STS2W9T"
"Date_first_available_at_Amazon_in":string"8 June 2019"
"Customer_Reviews":string"5.0 out of 5 stars 3 customer reviews"
"Amazon_Bestsellers_Rank":string"#19,259 in Clothing & Accessories (See Top 100 in Clothing & Accessories) #1793 in Women's Kurtas & Kurtis"
}
}
"""
